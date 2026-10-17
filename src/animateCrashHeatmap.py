"""Create a monthly crash-density GIF from the local NYC collision CSV.

Run: .venv/bin/python src/animateCrashHeatmap.py
Requires pandas, numpy, scipy, matplotlib and Pillow. The official borough
GeoJSON is cached in data/external/maps/nyc_boroughs.geojson.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/nyc-animation-mpl')
import argparse
import calendar
import json
from pathlib import Path
from math import cos, radians, ceil
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgba
from matplotlib.patches import Polygon
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/crash_timelapse'
BOUNDARY=ROOT/'data/external/maps/nyc_boroughs.geojson'
BOUNDARY_URL='https://data.cityofnewyork.us/resource/gthc-hcne.geojson?$limit=5'
WEST,SOUTH,EAST,NORTH=-74.26,40.49,-73.68,40.92
CELL_KM=.25
GAUSSIAN_SIGMA_CELLS=.4
DX=CELL_KM/(111.32*cos(radians(40.7)))
DY=CELL_KM/111.32
NX,NY=ceil((EAST-WEST)/DX),ceil((NORTH-SOUTH)/DY)
THEME='#BDE254'
BG,LAND,EDGE,TEXT,MUTED='#ffffff','#edf0e8','#687560','#355500','#46543d'


def load_months():
    grids={}; totals={}; mapped={}
    first=last=None; rows=0; invalid_dates=0
    for chunk in pd.read_csv(ROOT/'data/motor_vehicle_collisions.csv',
                             usecols=['CRASH DATE','LATITUDE','LONGITUDE'],chunksize=200000):
        dates=pd.to_datetime(chunk['CRASH DATE'],format='%m/%d/%Y',errors='coerce')
        rows+=len(chunk); invalid_dates+=int(dates.isna().sum())
        if dates.notna().any():
            lo,hi=dates.min(),dates.max()
            first=lo if first is None else min(first,lo)
            last=hi if last is None else max(last,hi)
        months=dates.dt.to_period('M')
        for month,count in months.value_counts().items():
            totals[month]=totals.get(month,0)+int(count)
        lat=pd.to_numeric(chunk.LATITUDE,errors='coerce')
        lon=pd.to_numeric(chunk.LONGITUDE,errors='coerce')
        valid=lat.between(SOUTH,NORTH)&lon.between(WEST,EAST)&dates.notna()
        data=pd.DataFrame({'month':months[valid],
                           'x':np.floor((lon[valid]-WEST)/DX).astype(int),
                           'y':np.floor((lat[valid]-SOUTH)/DY).astype(int)})
        for month,g in data.groupby('month'):
            grid=grids.setdefault(month,np.zeros((NY,NX),dtype=np.float64))
            np.add.at(grid,(g.y.to_numpy(),g.x.to_numpy()),1)
            mapped[month]=mapped.get(month,0)+len(g)
    start=first.to_period('M')+int(first.day!=1)
    end=last.to_period('M')-int(last.day!=calendar.monthrange(last.year,last.month)[1])
    months=list(pd.period_range(start,end,freq='M'))
    assert months and all(m in grids for m in months)
    density=[]; summary=[]
    for month in months:
        days=month.days_in_month
        grid=grids[month]
        assert int(grid.sum())==mapped[month]
        density.append(gaussian_filter(grid,sigma=GAUSSIAN_SIGMA_CELLS,mode='constant')/(days*CELL_KM**2))
        summary.append({'month':str(month),'days':days,'reported_crashes':totals[month],
                        'mapped_crashes':mapped[month], 'mapped_crashes_per_day':mapped[month]/days,
                        'mapped_share':mapped[month]/totals[month]})
    audit={'input_rows':rows,'invalid_date_rows':invalid_dates,'input_first_date':str(first.date()),
           'input_last_date':str(last.date()),'first_frame':str(months[0]),'last_frame':str(months[-1]),
           'frames':len(months),'cell_size_km':CELL_KM,'gaussian_sigma_km':round(GAUSSIAN_SIGMA_CELLS*CELL_KM,3),
           'metric':'All reported crashes with valid coordinates in the NYC map extent; smoothed counts per square km per day.',
           'exclusion':'Incomplete edge months excluded. Missing/out-of-extent coordinates omitted from map; monthly reported totals include them.',
           'comparison':'One fixed map extent and color scale across frames; power normalization gamma=0.5. No normalization by traffic exposure.',
           'boundary_source':BOUNDARY_URL,'boundary_accessed':'2026-10-17',
           'crash_source':'https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes'}
    return months,np.array(density),pd.DataFrame(summary),audit


def render(months,density,summary,audit,frame_ms):
    if not BOUNDARY.exists():
        BOUNDARY.parent.mkdir(parents=True,exist_ok=True)
        urlretrieve(BOUNDARY_URL,BOUNDARY)
    geo=json.loads(BOUNDARY.read_text())
    assert len(geo['features'])==5
    # Keep one scale across months, with rare peaks saturated for clearer contrast.
    maximum=6.0
    norm=PowerNorm(gamma=.5,vmin=0,vmax=maximum,clip=True)
    cmap=LinearSegmentedColormap.from_list('crash_warm',[
        (0,to_rgba(LAND,0)),(.06,to_rgba('#efba83',.65)),
        (.25,to_rgba('#e47732',.95)),(.55,to_rgba('#bf3c20',1)),
        (1,to_rgba('#731b16'))])
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':TEXT,'axes.labelcolor':MUTED,
                         'xtick.color':MUTED,'ytick.color':MUTED,'font.size':11})
    fig=plt.figure(figsize=(11,9),dpi=100,facecolor=BG)
    fig.text(.04,.955,'NYC crashes through time',fontsize=25,fontweight='bold')
    ax=fig.add_axes([.025,.235,.72,.70],facecolor=BG)
    for f in geo['features']:
        geometry=f['geometry']; polys=geometry['coordinates']
        if geometry['type']=='Polygon': polys=[polys]
        for polygon in polys:
            # Outer shorelines; interior rings are not material at this map scale.
            ax.add_patch(Polygon(polygon[0],closed=True,facecolor=LAND,edgecolor=EDGE,lw=.75,zorder=1))
    heat=ax.imshow(density[0],origin='lower',extent=[WEST,WEST+NX*DX,SOUTH,SOUTH+NY*DY],
                   cmap=cmap,norm=norm,interpolation='nearest',zorder=2)
    ax.set_xlim(WEST,EAST); ax.set_ylim(SOUTH,NORTH)
    ax.set_aspect(1/cos(radians(40.7))); ax.set_axis_off()
    for label,x,y in [('Manhattan',-74.027,40.795),('Bronx',-73.86,40.875),
                      ('Queens',-73.79,40.755),('Brooklyn',-73.96,40.63),
                      ('Staten\nIsland',-74.16,40.57)]:
        ax.text(x,y,label,ha='center',va='center',fontsize=11,color=TEXT,zorder=3,
                bbox={'facecolor':BG,'alpha':.78,'edgecolor':'none','pad':3})
    month_text=fig.text(.865,.79,'',ha='center',fontsize=26,fontweight='bold')
    fig.text(.865,.727,'CRASHES ON MAP',ha='center',fontsize=11,color=MUTED,fontweight='bold')
    count_text=fig.text(.865,.672,'',ha='center',fontsize=34,fontweight='bold')
    daily_text=fig.text(.865,.630,'',ha='center',fontsize=13,color=MUTED)
    cax=fig.add_axes([.775,.545,.18,.022],facecolor=LAND)
    cb=fig.colorbar(heat,cax=cax,orientation='horizontal',ticks=[0,maximum/4,maximum],extend='max')
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=10,length=0)
    cb.ax.set_xticklabels(['0',f'{maximum/4:g}',f'{maximum:g}+'])
    fig.text(.865,.480,'Smoothed crashes\nper km² per day',ha='center',fontsize=11,color=MUTED,linespacing=1.5)
    timeline=fig.add_axes([.08,.12,.85,.085],facecolor=BG)
    values=summary.mapped_crashes_per_day.to_numpy()
    x=np.arange(len(months))
    timeline.plot(x,values,color=MUTED,lw=2)
    dot,=timeline.plot([0],[values[0]],'o',color='#731b16',ms=8)
    line=timeline.axvline(0,color='#731b16',alpha=.9,lw=1.5)
    timeline.set_xlim(0,len(months)-1); timeline.set_ylim(0,values.max()*1.12)
    ticks=[i for i,m in enumerate(months) if m.month==1 and m.year in [2013,2016,2020,2023,2026]]
    timeline.set_xticks(ticks,[str(months[i].year) for i in ticks]); timeline.set_yticks([])
    for spine in timeline.spines.values(): spine.set_visible(False)
    timeline.tick_params(length=0)
    fig.text(.08,.22,'Mapped crashes per day across the whole city',fontsize=10,color=MUTED)
    fig.text(.04,.027,'Sources: NYC Motor Vehicle Collisions / NYC DCP borough boundaries · Reporting and travel changed in 2020',fontsize=9,color=MUTED)
    frames=[]; palette=None
    for i,month in enumerate(months):
        heat.set_data(density[i]); row=summary.iloc[i]
        month_text.set_text(month.strftime('%b %Y'))
        count_text.set_text(f'{int(row.mapped_crashes):,}')
        daily_text.set_text(f'{row.mapped_crashes_per_day:,.0f} crashes / day')
        dot.set_data([i],[values[i]]); line.set_xdata([i,i])
        fig.canvas.draw()
        rgb=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
        if palette is None:
            # The first frame includes the full colorbar, yielding one stable palette for the GIF.
            palette=rgb.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
        frame=rgb.quantize(palette=palette,dither=Image.Dither.NONE)
        frames.append(frame)
        if i in [0,len(months)-1] or str(month)=='2020-04':
            rgb.save(OUT/f'preview_{month}.png')
        if (i+1)%24==0: print(f'Rendered {i+1}/{len(months)} months',flush=True)
    durations=[frame_ms]*len(frames); durations[0]=1000; durations[-1]=1800
    target=OUT/'nyc_crashes_monthly.gif'
    frames[0].save(target,save_all=True,append_images=frames[1:],duration=durations,
                   loop=0,optimize=False,disposal=2)
    plt.close(fig)
    with Image.open(target) as gif:
        assert gif.n_frames==len(months)
        assert gif.info.get('loop')==0
        audit.update({'gif_frames_verified':gif.n_frames,'fixed_color_maximum':maximum,
                      'theme_color':THEME,'theme_applies_to':'Green theme with dark green text for contrast',
                      'text_color':TEXT,'secondary_text_color':MUTED,
                      'background_color':BG,'heatmap_interpolation':'nearest',
                      'heatmap_palette':'High-contrast amber, burnt orange, and deep brick red on pale gray-green land.',
                      'color_scale_rule':'Fixed 0–6+ scale across all months; values above 6 saturated.',
                      'density_global_peak':float(density.max()),
                      'comparison':'One fixed map extent and capped color scale across frames; power normalization gamma=0.5. No normalization by traffic exposure.',
                      'frame_duration_ms':frame_ms,'first_pause_ms':1000,'last_pause_ms':1800,
                      'gif_bytes':target.stat().st_size})
    (OUT/'metadata.json').write_text(json.dumps(audit,indent=2))
    print(f'Saved {target} ({target.stat().st_size/1e6:.1f} MB)',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame-ms',type=int,default=180,help='Milliseconds per monthly frame (default 180).')
    args=parser.parse_args()
    if args.frame_ms<20: parser.error('--frame-ms must be at least 20')
    OUT.mkdir(parents=True,exist_ok=True)
    months,density,summary,audit=load_months()
    summary.to_csv(OUT/'monthly_counts.csv',index=False)
    print(f'Prepared {len(months)} full months: {months[0]} to {months[-1]}',flush=True)
    render(months,density,summary,audit,args.frame_ms)
    (OUT/'README.md').write_text('# NYC crash heatmap timelapse\n\n'
        'Open `nyc_crashes_monthly.gif`. Each frame shows one complete calendar month. '
        'The fixed color scale measures spatially smoothed reported crashes per square kilometer per day; '
        'month length is accounted for, traffic exposure is not. All vehicle-involved crashes are included. '
        'The first and last months pause for readability.\n\n'
        'Run `.venv/bin/python src/animateCrashHeatmap.py` to regenerate. '
        'Use `--frame-ms 300` for slower playback. Requires numpy, pandas, scipy, matplotlib and Pillow. '
        'The cached borough outlines allow offline reruns.\n\n'
        'Frames exclude incomplete edge months and omit crashes missing usable coordinates. '
        'The mapped share varies over time; reporting rules and travel changed in 2020. '
        'Colors indicate burden, not risk per trip or a causal trend. Smoothing uses 250 m cells and a 100 m Gaussian sigma. '
        'Nearest-neighbor rendering keeps individual map cells sharp. '
        'The color scale uses a square-root transformation to keep lower concentrations visible.\n\n'
        'The fixed color scale runs from 0 to 6+ crashes per km² per day. '
        'Values above 6 use the hottest color.\n\n'
        'Sources: [NYC crashes](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes); '
        '[NYC DCP borough boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne). '
        'Boundaries retrieved October 17, 2026. See metadata.json for coverage and export checks.\n')

if __name__=='__main__': main()
