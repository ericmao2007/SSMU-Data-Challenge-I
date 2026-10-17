"""Render two minimal monthly NYC GIFs from the local crash data.

Run: .venv/bin/python src/animateCrashMinimal.py
Counts include all reports; maps omit missing/out-of-extent coordinates.
The people metric is persons injured plus persons killed, not all involved.
"""
import argparse
import json

from animateCrashHeatmap import (
    ROOT, BOUNDARY, BG, LAND, EDGE, TEXT, MUTED,
    WEST, SOUTH, EAST, NORTH, NX, NY, DX, DY, CELL_KM,
    np, pd, plt, Image, Polygon, PowerNorm, LinearSegmentedColormap,
    gaussian_filter, GAUSSIAN_SIGMA_CELLS, cos, radians, load_months,
)

OUT = ROOT / 'outputs/crash_timelapse_minimal'


def load_people(months):
    grids = {m: np.zeros((NY, NX)) for m in months}
    totals = {m: 0 for m in months}
    mapped = {m: 0 for m in months}
    columns = ['CRASH DATE', 'LATITUDE', 'LONGITUDE',
               'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED']
    for chunk in pd.read_csv(ROOT / 'data/motor_vehicle_collisions.csv',
                             usecols=columns, chunksize=200000):
        periods = pd.to_datetime(chunk['CRASH DATE'], format='%m/%d/%Y',
                                 errors='coerce').dt.to_period('M')
        weights = chunk[columns[-2:]].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
        lat = pd.to_numeric(chunk.LATITUDE, errors='coerce')
        lon = pd.to_numeric(chunk.LONGITUDE, errors='coerce')
        for month, value in weights.groupby(periods).sum().items():
            if month in totals:
                totals[month] += int(value)
        valid = lat.between(SOUTH, NORTH) & lon.between(WEST, EAST) & periods.isin(months)
        points = pd.DataFrame({'month': periods[valid], 'weight': weights[valid],
                               'x': np.floor((lon[valid] - WEST) / DX).astype(int),
                               'y': np.floor((lat[valid] - SOUTH) / DY).astype(int)})
        for month, group in points.groupby('month'):
            np.add.at(grids[month], (group.y.to_numpy(), group.x.to_numpy()), group.weight.to_numpy())
            mapped[month] += int(group.weight.sum())
    for month in months:
        assert int(grids[month].sum()) == mapped[month] <= totals[month]
    density = np.array([gaussian_filter(grids[m], sigma=GAUSSIAN_SIGMA_CELLS, mode='constant') /
                        (m.days_in_month * CELL_KM ** 2) for m in months])
    return density, [totals[m] for m in months], [mapped[m] for m in months]


def render(months, density, counts, metric, frame_ms):
    geo = json.loads(BOUNDARY.read_text())
    maximum = float(np.ceil(density.max()))
    cmap = LinearSegmentedColormap.from_list('minimal_glow', [
        (0, (.14, .20, .27, 0)), (.06, (.94, .73, .51, .65)),
        (.25, (.89, .47, .20, .95)), (.55, (.75, .24, .13, 1)), (1, (.45, .11, .09, 1))])
    fig = plt.figure(figsize=(8, 8), dpi=100, facecolor=BG)
    ax = fig.add_axes([.02, .01, .96, .81], facecolor=BG)
    for feature in geo['features']:
        geometry = feature['geometry']
        polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
        for polygon in polygons:
            ax.add_patch(Polygon(polygon[0], closed=True, facecolor=LAND,
                                 edgecolor=EDGE, lw=.75, zorder=1))
    heat = ax.imshow(density[0], origin='lower', extent=[WEST, WEST + NX * DX, SOUTH, SOUTH + NY * DY],
                     cmap=cmap, norm=PowerNorm(gamma=.5, vmin=0, vmax=maximum),
                     interpolation='nearest', zorder=2)
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)
    ax.set_aspect(1 / cos(radians(40.7)))
    ax.set_axis_off()
    title = fig.text(.5, .942, '', ha='center', color=TEXT, fontsize=21, fontweight='bold')
    count = fig.text(.5, .878, '', ha='center', color=MUTED, fontsize=18)
    frames = []
    palette = None
    for i, month in enumerate(months):
        title.set_text(f'NYC crashes · {month.strftime("%b %Y")}')
        count.set_text(f'{counts[i]:,} ' + ('crashes' if metric == 'crashes' else 'people injured or killed'))
        heat.set_data(density[i])
        fig.canvas.draw()
        rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        if palette is None:
            # Include the full heat ramp in the shared palette, even if absent in frame 1.
            ramp = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            sample = np.asarray(rgb).copy()
            sample[-24:, :, :] = ramp[np.linspace(0, 255, sample.shape[1]).astype(int)][None, :, :]
            palette = Image.fromarray(sample).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
        if i == 0 or str(month) == '2020-04' or i == len(months) - 1:
            rgb.save(OUT / f'{metric}_{month}.png')
    durations = [frame_ms] * len(frames)
    durations[0], durations[-1] = 1000, 1800
    target = OUT / f'nyc_{metric}_monthly_minimal.gif'
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=2)
    plt.close(fig)
    with Image.open(target) as gif:
        assert gif.n_frames == len(months) and gif.info['loop'] == 0
        assert gif.size == (800, 800)
    print(f'Saved {target}', flush=True)
    return {'file': target.name, 'frames': len(months), 'fixed_color_maximum': maximum,
            'text_color': TEXT, 'secondary_text_color': MUTED,
            'heatmap_interpolation': 'nearest',
            'heatmap_palette': 'High-contrast amber, burnt orange, and deep brick red.',
            'bytes': target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame-ms', type=int, default=180)
    args = parser.parse_args()
    if args.frame_ms < 20:
        parser.error('--frame-ms must be at least 20')
    OUT.mkdir(parents=True, exist_ok=True)
    months, crash_density, summary, audit = load_months()
    people_density, people_counts, mapped_people = load_people(months)
    summary['people_injured_or_killed'] = people_counts
    summary['mapped_people_injured_or_killed'] = mapped_people
    summary.to_csv(OUT / 'monthly_counts.csv', index=False)
    audit['exports'] = [render(months, crash_density, summary.reported_crashes.tolist(), 'crashes', args.frame_ms),
                        render(months, people_density, people_counts, 'people', args.frame_ms)]
    audit['count_definition'] = 'Citywide monthly reported crashes; people = NUMBER OF PERSONS INJURED + NUMBER OF PERSONS KILLED. Counts include reports without map coordinates; maps omit those reports.'
    audit['map_definition'] = 'Separate fixed color scales across time: smoothed crashes or people injured/killed per km² per day, respectively. Colors are not comparable between the two GIFs.'
    (OUT / 'metadata.json').write_text(json.dumps(audit, indent=2))
    (OUT / 'README.md').write_text(
        '# Minimal NYC monthly crash GIFs\n\n'
        'Each GIF shows only a map, a title with the month, and the monthly count. '
        'The crashes GIF counts reported collisions. The people GIF counts persons injured plus persons killed; '
        'the source does not count everyone involved. Counts include all reports, while maps require valid coordinates. '
        'Only complete edge months are included. Each map uses its own fixed color scale across time, '
        'with the same smoothing and daily density normalization as the original animation. '
        'Colors represent burden, not risk per trip. Reporting and travel changed in 2020.\n\n'
        'Run `.venv/bin/python src/animateCrashMinimal.py` to regenerate. '
        'See monthly_counts.csv and metadata.json for counts and export checks.\n')


if __name__ == '__main__':
    main()
