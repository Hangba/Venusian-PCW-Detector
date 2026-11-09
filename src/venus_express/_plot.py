import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import datetime

from venus_express.paths import DATA_DIR
from venus_express.mpl_style import apply_matplotlib_style


apply_matplotlib_style(font_size=12)

class BasePlotter:
    """
    Generic plotting base class that quickly aligns multiple variables in one figure with UTC ticks.
    """
    def __init__(self, trange, fs=1.0):
        """
        Initialize the plotter.
        :param trange: Time range, e.g. ['2006-05-08/03:30:00', '2006-05-08/03:34:59']
        :param fs: Sampling frequency (Hz)
        """
        self.trange = trange
        self.fs = fs
        self.t_start = pd.to_datetime(trange[0], utc=True)
        self.t_end = pd.to_datetime(trange[1], utc=True)
        self.crossing_times = self._read_crossing_times(str(DATA_DIR / 'Venus_Express_Crossing_v1.csv'))

    def _read_crossing_times(self, crossing_file):
        import os
        if not os.path.exists(crossing_file):
            return None

        try:
            from venus_express.io.crossings import load_crossing_table
            df = load_crossing_table(crossing_file)
            df = df.loc[df.crossing_order_valid].copy()
            paired = df.crossing_record_kind.eq('two_boundary')
            df.loc[paired, 'SW_Start'] = df.loc[paired, 'bs_out']
            df.loc[paired, 'MSH_End'] = pd.NaT

            # Use UTC to avoid local-timezone interference
            df['SW_End'] = pd.to_datetime(df['SW_End'], utc=True)
            df['MSH_Start'] = pd.to_datetime(df['MSH_Start'], utc=True)
            df['MSH_End'] = pd.to_datetime(df['MSH_End'], utc=True)
            df['SW_Start'] = pd.to_datetime(df['SW_Start'], utc=True)

            mask = (
                ((df['SW_End'] >= self.t_start) & (df['SW_End'] <= self.t_end)) |
                ((df['MSH_Start'] >= self.t_start) & (df['MSH_Start'] <= self.t_end)) |
                ((df['MSH_End'] >= self.t_start) & (df['MSH_End'] <= self.t_end)) |
                ((df['SW_Start'] >= self.t_start) & (df['SW_Start'] <= self.t_end))
            )
            df_filtered = df[mask]

            if df_filtered.empty:
                return None

            crossing_times = {
                'SW_End': [], 'MSH_Start': [], 'MSH_End': [], 'SW_Start': []
            }

            for _, row in df_filtered.iterrows():
                for col in ['SW_End', 'MSH_Start', 'MSH_End', 'SW_Start']:
                    time_val = row[col]
                    if pd.notna(time_val) and self.t_start <= time_val <= self.t_end:
                        crossing_times[col].append(time_val)

            return crossing_times if any(crossing_times.values()) else None

        except Exception as e:
            print(f"读取crossing文件时出错: {e}")
            return None

    def tplot(self, variables, title = "", figsize=(14, 12), show_fig = True):
        """
        Plot the given list of variable dictionaries, stacked top to bottom and
        sharing one X axis (UT time).
        Automatically detects line plots vs. spectra (a 'freqs' key or 2D data
        with a large second dimension is treated as a spectrum).

        :param variables: List of variable dictionaries, each containing:
            - data: np.ndarray, required, the main data
            - times: np.ndarray or pd.Series, timestamp sequence, usually the x axis
            - freqs: np.ndarray, optional, frequency/Y-axis ticks for spectra
            - ylabel, zlabel, title, cmap, vmin, vmax, labels, colors, etc.
        :param figsize: Figure size
        """
        n_plots = len(variables)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        fig.suptitle(title, fontsize=16)  # Set the overall title
        if n_plots == 1:
            axes = [axes]

        for i, var_info in enumerate(variables):
            ax = axes[i]
            data = var_info['data']
            times = var_info.get('times', None)

            # Use matplotlib's standard time format for X-axis alignment
            if times is not None:
                x_times = times
            else:
                x_times = np.arange(len(data))

            # Detect spectra: a 'freqs' key, or 2D data with a large shape[1] (> 4)
            is_spectra = 'freqs' in var_info
            if not is_spectra and data.ndim == 2 and data.shape[1] > 4:
                is_spectra = True

            # Reserve colorbar space on the right of every subplot to keep axes aligned
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.1)

            if is_spectra:
                # Draw a heatmap / spectrogram
                freqs = var_info.get('freqs', np.arange(data.shape[1] if data.shape[0] == len(x_times) else data.shape[0]))
                cmap_choice = var_info.get('cmap', 'viridis')
                vmin = var_info.get('vmin', None)
                vmax = var_info.get('vmax', None)

                # data must match the grid; if it is (time, frequency), transpose to (frequency, time) for pcolormesh
                Z = data.T if data.shape[0] == len(x_times) else data

                # pcolormesh requires a meshgrid
                if len(freqs) == Z.shape[0] and len(x_times) == Z.shape[1]:
                    X, Y = np.meshgrid(x_times, freqs)
                    mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap_choice, vmin=vmin, vmax=vmax)
                else: # Avoid errors when shapes do not match exactly
                    mesh = ax.pcolormesh(x_times, freqs, Z, shading='auto', cmap=cmap_choice, vmin=vmin, vmax=vmax)

                cbar = plt.colorbar(mesh, cax=cax)
                if 'zlabel' in var_info:
                    cbar.set_label(var_info['zlabel'])

                # Add extra overlay curves (e.g. local proton gyrofrequency f_cp)
                if 'overlay_lines' in var_info:
                    for line_dict in var_info['overlay_lines']:
                        overlay_times = line_dict.get('times', x_times)
                        ax.plot(overlay_times, line_dict['data'], color=line_dict.get('color', 'w'),
                                linestyle=line_dict.get('linestyle', '--'), linewidth=line_dict.get('linewidth', 1.0),
                                label=line_dict.get('label', None))
                    if any(line_dict.get('label') for line_dict in var_info['overlay_lines']):
                        ax.legend(loc='upper right')

                # Use a log axis when yscale is 'log'
                if var_info.get('yscale') == 'log':
                    ax.set_yscale('log')

                ax.set_ylabel(var_info.get('ylabel', 'Frequency (Hz)'))
                ax.set_title(var_info.get('title', 'Spectrogram'))

            else:
                # Draw a plain 1D curve / multi-line plot
                cax.axis('off') # Line plots need no colorbar: hide it but keep the space so the axes stay aligned
                labels = var_info.get('labels', ['Data'])
                colors = var_info.get('colors', [None] * (data.shape[1] if data.ndim > 1 else 1))

                if data.ndim == 1:
                    ax.plot(x_times, data, label=labels[0] if len(labels)>0 else None, color=colors[0] if len(colors)>0 else None, linewidth=0.8)
                else:
                    for j in range(data.shape[1]):
                        l = labels[j] if j < len(labels) else f"component_{j}"
                        c = colors[j] if j < len(colors) else None
                        ax.plot(x_times, data[:, j], label=l, color=c, linewidth=0.8)

                ax.set_ylabel(var_info.get('ylabel', 'Value'))
                ax.set_title(var_info.get('title', 'Time Series'))
                if var_info.get('labels'):
                    ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

            # Apply a common y-axis range when ylim=[ymin, ymax] is given
            if 'ylim' in var_info:
                ax.set_ylim(var_info['ylim'])

        # Add dashed crossing-time markers to all subplots
        if getattr(self, 'crossing_times', None):
            boundary_styles = {
                'SW_End': {'color': 'blue', 'linestyle': '--', 'linewidth': 1.5, 'label': 'SW End', 'alpha': 0.7},
                'MSH_Start': {'color': 'green', 'linestyle': '--', 'linewidth': 1.5, 'label': 'MSH Start', 'alpha': 0.7},
                'MSH_End': {'color': 'orange', 'linestyle': '--', 'linewidth': 1.5, 'label': 'MSH End', 'alpha': 0.7},
                'SW_Start': {'color': 'red', 'linestyle': '--', 'linewidth': 1.5, 'label': 'SW Start', 'alpha': 0.7}
            }
            has_line = False
            for ax in axes:
                for boundary_type, times_list in self.crossing_times.items():
                    for t_val in times_list:
                        style = boundary_styles[boundary_type]
                        ax.axvline(x=t_val, color=style['color'],
                                   linestyle=style['linestyle'],
                                   linewidth=style['linewidth'],
                                   alpha=style['alpha'])
                        has_line = True

            # Add a legend on the top layer of the figure
            if has_line:
                from matplotlib.lines import Line2D
                handles, labels_list = [], []
                for boundary_type, times_list in self.crossing_times.items():
                    if times_list:
                        style = boundary_styles[boundary_type]
                        handles.append(Line2D([0], [0], color=style['color'],
                                             linestyle=style['linestyle'],
                                             linewidth=style['linewidth']))
                        labels_list.append(style['label'])
                if handles:
                    axes[0].legend(handles, labels_list, loc='upper right', fontsize=9, framealpha=0.8)

        # Apply uniform X-axis time formatting
        axes[-1].set_xlim(self.t_start, self.t_end)
        axes[-1].set_xlabel('Universal Time')

        # Define the UT display format (HH:MM:SS)
        myFmt = mdates.DateFormatter('%H:%M:%S')
        axes[-1].xaxis.set_major_formatter(myFmt)

        # Check whether the range spans multiple days
        if self.t_start.date() == self.t_end.date():
            # Same day: show the date in the bottom-right corner
            date_str = self.t_start.strftime('%Y-%m-%d')
            fig.text(0.99, 0.01, date_str, ha='right', va='bottom', fontsize=10, transform=fig.transFigure)
            # Add regular minor ticks
            from matplotlib.ticker import AutoMinorLocator
            axes[-1].xaxis.set_minor_locator(AutoMinorLocator())
        else:
            # Multi-day: mark minor ticks and dates at each day boundary (00:00); no bottom-right date
            tz = datetime.timezone.utc
            axes[-1].xaxis.set_minor_locator(mdates.DayLocator(tz=tz))
            axes[-1].xaxis.set_minor_formatter(mdates.DateFormatter('\n%Y-%m-%d', tz=tz))
            axes[-1].tick_params(axis='x', which='minor', length=15, width=1.5, color='black', labelsize=10, labelcolor='black', labelbottom=True)

        fig.autofmt_xdate(rotation=0, ha='center')  # Automatically adjust label rotation


        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        self.variables = variables  # Store variables for later figure saving
        self.figsize = figsize  # Store figsize for later figure saving
        if show_fig:
            self.fig = fig  # Store the fig object for later figure saving
            plt.show()
        else:
            self.fig = fig  # Store the fig object for later figure saving
            plt.close(fig)  # Close the figure without destroying the fig object



    def save(self, filename, dpi=300):
        """
        Save the current figure to a file.
        :param filename: File name, e.g. 'output.png'
        :param dpi: Figure resolution, defaults to 300
        """
        self.fig.savefig("runs/legacy_output/" + filename, dpi=dpi, bbox_inches='tight')
