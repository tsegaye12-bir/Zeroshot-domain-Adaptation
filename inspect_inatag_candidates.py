#!/usr/bin/env python3
"""Inspect selected public iNatAg-mini species archives for geographic split feasibility."""
import concurrent.futures
import csv
import io
import json
import os
import pathlib
import zipfile
from collections import Counter, defaultdict

import requests
from shapely.geometry import Point, shape

BASE = pathlib.Path('/home/ubuntu/research')
ARCHIVES = BASE / 'archives'
ARCHIVES.mkdir(parents=True, exist_ok=True)
META_PATH = pathlib.Path('/home/ubuntu/AgML/agml/_assets/iNatAg-mini_public_datasources.json')
SPECIES = [
    'amaranthus_retroflexus', 'chenopodium_album', 'taraxacum_officinale',
    'capsella_bursa-pastoris', 'portulaca_oleracea', 'solanum_nigrum',
    'convolvulus_arvensis', 'ambrosia_artemisiifolia', 'plantago_major',
    'raphanus_raphanistrum', 'setaria_viridis', 'cyperus_rotundus',
    'galinsoga_parviflora', 'erigeron_canadensis', 'xanthium_strumarium',
    'zea_mays', 'phaseolus_vulgaris', 'triticum_aestivum', 'glycine_max',
]


def load_continents():
    url = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson'
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    fc = r.json()
    return [(shape(f['geometry']), f['properties'].get('CONTINENT')) for f in fc['features'] if f.get('geometry')]


def download_species(species):
    path = ARCHIVES / f'{species}.zip'
    if not path.exists():
        url = f'https://huggingface.co/datasets/Project-AgML/iNatAg-mini/resolve/main/{species}.zip?download=true'
        r = requests.get(url, timeout=240)
        r.raise_for_status()
        path.write_bytes(r.content)
    return species, path


def get_continent(lon, lat, polygons):
    if lon is None or lat is None:
        return 'Missing'
    try:
        p = Point(float(lon), float(lat))
    except (TypeError, ValueError):
        return 'Missing'
    if not (-90 <= p.y <= 90 and -180 <= p.x <= 180):
        return 'Invalid'
    hits = [continent for geom, continent in polygons if geom.covers(p)]
    hits = [x for x in hits if x and x != 'Antarctica']
    return hits[0] if hits else 'Unassigned'


def main():
    meta = json.loads(META_PATH.read_text())
    present = [s for s in SPECIES if f'iNatAg-mini/{s}' in meta]
    print('candidate_species_found', len(present), 'of', len(SPECIES))
    for s in present:
        md = meta[f'iNatAg-mini/{s}']
        print('metadata', s, 'expected_images', md.get('n_images'), 'species_label', list(md.get('classes', {}).values()), 'common_names', md.get('extra_metadata', {}).get('common_name', [])[:2])

    polygons = load_continents()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        downloads = list(pool.map(download_species, present))
    for species, path in downloads:
        with zipfile.ZipFile(path) as zf:
            ann_name = next(n for n in zf.namelist() if n.lower().endswith('annotations.json'))
            ann = json.loads(zf.read(ann_name))
        for a in ann:
            continent = get_continent(a.get('longitude'), a.get('latitude'), polygons)
            rows.append({
                'species': species,
                'photo_id': a.get('photo_id'),
                'continent': continent,
                'crop_weed': a.get('crop/weed'),
                'observed_on': a.get('observed_on'),
                'user': a.get('user'),
                'image_url': a.get('image_url'),
                'image_path': a.get('save_path'),
            })

    out_csv = BASE / 'inatag_mini_candidate_manifest.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    counts = defaultdict(Counter)
    for r in rows:
        counts[r['species']][(r['continent'], r['crop_weed'])] += 1
    for species in present:
        print('counts', species, dict(counts[species]))
    print('manifest', out_csv)
    print('archive_dir', ARCHIVES)

if __name__ == '__main__':
    main()
