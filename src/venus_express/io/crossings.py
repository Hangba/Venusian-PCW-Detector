"""Boundary-crossing metadata readers."""

import os
from datetime import datetime

import pandas as pd
import numpy as np

from venus_express.paths import DATA_DIR


def load_crossing_table(crossing_file):
    """Normalize full legacy records and confirmed v2 bow-shock pairs.

    Appended v2 rows use SW_End / MSH_End for inbound / outbound bow shock.
    Their missing source orbit/periapsis fields stay missing. An interval
    midpoint is used only to associate nearby event times, never as a measured
    periapsis. The original boundary columns are not overwritten.
    """
    table=pd.read_csv(crossing_file)
    for c in ['PeriTime(UTC)','SW_End','MSH_Start','MSH_End','SW_Start']:
        if c not in table:table[c]=pd.NaT
        table[c]=pd.to_datetime(table[c],utc=True,format='mixed',errors='coerce').astype('datetime64[ns, UTC]')
    if 'Orbit_NO' not in table:table['Orbit_NO']=pd.NA
    table['Orbit_NO']=pd.to_numeric(table.Orbit_NO,errors='coerce').astype('Int64')
    full=table[['SW_End','MSH_Start','MSH_End','SW_Start']].notna().all(axis=1)
    paired=table.SW_End.notna()&table.MSH_End.notna()&table.MSH_Start.isna()&table.SW_Start.isna()
    msh_only=table.MSH_Start.notna()&table.MSH_End.notna()&table.SW_End.isna()&table.SW_Start.isna()
    table['crossing_record_id']=np.arange(1,len(table)+1)
    table['crossing_source_row']=table.crossing_record_id+1
    table['crossing_record_kind']=np.select([full,paired,msh_only],['four_boundary','two_boundary','msh_interval_only'],default='incomplete')
    table['bs_in']=table.SW_End
    table['bs_out']=table.SW_Start.where(~paired,table.MSH_End)
    table['sheath_start']=table.MSH_Start.where(~paired,table.SW_End)
    table['sheath_end']=table.MSH_End
    table['crossing_order_valid']=(
        (full&(table.SW_End<=table.MSH_Start)&(table.MSH_Start<table.MSH_End)&(table.MSH_End<=table.SW_Start))|
        (paired&table.SW_End.lt(table.MSH_End))|
        (msh_only&table.MSH_Start.lt(table.MSH_End)))
    table['crossing_reference']=table['PeriTime(UTC)']
    fallback=(table.sheath_start+(table.sheath_end-table.sheath_start)/2)
    table['crossing_reference']=table.crossing_reference.fillna(fallback)
    table['crossing_reference_kind']=np.where(table['PeriTime(UTC)'].notna(),'measured_periapsis','boundary_midpoint')
    table.attrs['crossing_version']='v2' if paired.any() else 'v1'
    return table


def read_crossing_times(date, crossing_file=str(DATA_DIR / "Venus_Express_Crossing_v1.csv")):
    """Read crossing times for a single UTC date."""
    if not os.path.exists(crossing_file):
        return None

    try:
        df = load_crossing_table(crossing_file)
        df = df.loc[df.crossing_order_valid].copy()
        # Return canonical bow-shock meanings for v2 pairs. Do not invent a
        # finite shock-transition duration when only a single time is recorded.
        paired=df.crossing_record_kind.eq('two_boundary')
        df.loc[paired,'SW_Start']=df.loc[paired,'bs_out']
        df.loc[paired,'MSH_End']=pd.NaT
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        mask = (
            (df["SW_End"].dt.date == date_obj)
            | (df["MSH_Start"].dt.date == date_obj)
            | (df["MSH_End"].dt.date == date_obj)
            | (df["SW_Start"].dt.date == date_obj)
        )
        df_date = df[mask]
        if df_date.empty:
            return None

        crossing_times = {
            "SW_End": [],
            "MSH_Start": [],
            "MSH_End": [],
            "SW_Start": [],
        }
        for _, row in df_date.iterrows():
            for col in ["SW_End", "MSH_Start", "MSH_End", "SW_Start"]:
                time_val = row[col]
                if pd.notna(time_val) and time_val.date() == date_obj:
                    crossing_times[col].append(time_val.to_pydatetime().replace(tzinfo=None))

        return crossing_times if any(crossing_times.values()) else None
    except Exception as exc:
        print(f"读取 crossing 文件时出错: {exc}")
        return None
