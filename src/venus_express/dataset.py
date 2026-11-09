'''
Process Venus Express data.
'''
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd

from venus_express.paths import VEX_DATA_DIR
from venus_express.physics.fac import convert_to_fac_coordinates
from venus_express.signal.filters import butterworth_filter


class VenusExpressDataset:
    def __init__(self, data_path, trange=None):
        self.data_path = data_path
        self.trange = trange
        self.data_dict = None
        self.trange_list = []
        self.dataframe = pd.DataFrame(
            columns=['time_ut', 'ut_seconds', 'Bx', 'By', 'Bz', 'Bt', 'pos_x', 'pos_y', 'pos_z']
        )
        if trange:
            self.load_data()
        self.dt = 1.0  # Default sampling interval in seconds
        self.fs = 1.0 / self.dt  # Sampling frequency in Hz

    @staticmethod
    def _get_date_list(trange):
        """Build a date list from the time range."""
        start_ts, end_ts = (pd.to_datetime(t, utc=True) for t in trange)
        start, end = start_ts.date(), end_ts.date()
        dates = []
        curr = start
        while curr <= end:
            dates.append(curr.strftime('%Y-%m-%d'))
            curr += timedelta(days=1)
        return dates

    @staticmethod
    def _read_single_file(file_path):
        """Read a single file."""
        try:
            df = pd.read_csv(
                file_path, sep=r'\s+', comment='#', header=None,
                names=['UTC', 'OBT', 'BIS_X', 'BIS_Y', 'BIS_Z', 'BIS_T',
                       'BOS_X', 'BOS_Y', 'BOS_Z', 'BOS_T',
                       'POS_X', 'POS_Y', 'POS_Z', 'POS_R', 'FLAG']
            )

            valid_mask = (df['BOS_X'] != 99999.999) & (df['BOS_Y'] != 99999.999) & (df['BOS_Z'] != 99999.999)
            df_valid = df.loc[valid_mask, ['UTC', 'BOS_X', 'BOS_Y', 'BOS_Z', 'BOS_T', 'POS_X', 'POS_Y', 'POS_Z']].copy()
            df_valid.columns = ['UTC', 'Bx', 'By', 'Bz', 'Bt', 'pos_x', 'pos_y', 'pos_z']
            return df_valid
        except Exception as e:
            print(f"读取 {file_path} 出错: {e}")
            return None


    def load_data(self):
        """Load all data files in the given time range."""
        date_list = self._get_date_list(self.trange)
        file_patterns = [os.path.join(self.data_path, f'BAM_{date_str}T*.dat') for date_str in date_list]
        files = sorted([f for pattern in file_patterns for f in glob.glob(pattern)])

        # Read files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            dfs = [df for df in executor.map(self._read_single_file, files) if df is not None]

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            # Use a single UTC (UT) time axis
            combined['time_ut'] = pd.to_datetime(combined['UTC'], utc=True)
            combined = combined.sort_values('time_ut').reset_index(drop=True)
            combined['ut_seconds'] = combined['time_ut'].map(lambda t: t.timestamp()).astype(np.float64)
            self.dataframe = combined[['time_ut', 'ut_seconds', 'Bx', 'By', 'Bz', 'Bt', 'pos_x', 'pos_y', 'pos_z']]
        else:
            # Keep an empty table when no data is available so downstream code does not crash on missing attributes
            self.dataframe = pd.DataFrame(
                columns=['time_ut', 'ut_seconds', 'Bx', 'By', 'Bz', 'Bt', 'pos_x', 'pos_y', 'pos_z']
            )

    def tofac(self, filter=butterworth_filter):
        """Convert to FAC coordinates."""
        if self.dataframe is None or self.dataframe.empty:
            # Preserve the column structure so upstream code can access columns uniformly
            for col in ['B_fac_x', 'B_fac_y', 'B_fac_z']:
                if col not in self.dataframe.columns:
                    self.dataframe[col] = pd.Series(dtype=np.float64)
            return

        if not self.dataframe['ut_seconds'].is_monotonic_increasing:
            self.dataframe = self.dataframe.sort_values('ut_seconds').reset_index(drop=True)

        bx = self.dataframe['Bx'].values
        by = self.dataframe['By'].values
        bz = self.dataframe['Bz'].values
        fac_payload = convert_to_fac_coordinates(
            {"Bx": bx, "By": by, "Bz": bz},
            cutoff_freq=0.01,
            fs=self.fs,
            filter_fn=filter,
        )

        self.dataframe['B_fac_x'] = fac_payload['B_fac_x']
        self.dataframe['B_fac_y'] = fac_payload['B_fac_y']
        self.dataframe['B_fac_z'] = fac_payload['B_fac_z']
        for component in ('Bx_bg', 'By_bg', 'Bz_bg'):
            self.dataframe[component] = fac_payload[component]



if __name__ == "__main__":
    dataset = VenusExpressDataset(
        data_path=str(VEX_DATA_DIR),
        trange=["2006-05-08/03:30:00", "2006-05-08/03:33:59"],
    )
    dataset.tofac()
    print(dataset.dataframe.head())
