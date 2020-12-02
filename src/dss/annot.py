"""[summary]

TODO: From/to traces
"""
import numpy as np
import xarray as xr
import pandas as pd
from collections import UserDict
from typing import Optional, List, Dict, Any, Union

class Events(UserDict):

    def __init__(self, data: Optional[Dict[str, List[float]]] = None, categories: Optional[Dict[str, str]] = None,
                 add_names_from_categories: bool = True):
        """[summary]

        Args:
            data: dict or Events
            categories (dict[str: str]): name, category mapping

        """
        if data is None:
            data = dict()

        super().__init__(data)

        for key, val in self.items():
            val = np.array(val)
            if val.ndim==1:
                val = val[:, np.newaxis]
            if val.shape[1]==1:
                val = np.concatenate((val, val), axis=1)

            self.data[key] = val
        self.categories = self._infer_categories()

        # drop nan
        self._drop_nan()

        # preserve cats from input
        if hasattr(data, 'categories'):
            for name, cat in data.categories.items():
                if name in self:  # update only existing keys
                    self.categories[name] = cat

        # update cats from arg
        if categories is not None:
            for name, cat in categories.items():
                if name in self:  # update only existing keys
                    self.categories[name] = cat
                elif add_names_from_categories:
                    self.add_name(name=name, category=cat)

    @classmethod
    def from_df(cls, df, possible_event_names=[]):
        return cls.from_lists(df.name.values,
                              df.start_seconds.values.astype(np.float),
                              df.stop_seconds.values.astype(np.float),
                              possible_event_names)

    @classmethod
    def from_lists(cls, names, start_seconds, stop_seconds, possible_event_names=[]):
        unique_names = list(set(names))
        unique_names.extend(possible_event_names)
        dct = {name: [] for name in unique_names}

        for name, start_second, stop_second in zip(names, start_seconds, stop_seconds):
            dct[name].append([start_second, stop_second])

        return cls(dct)

    @classmethod
    def from_dataset(cls, ds):
        start_seconds = np.array(ds.event_times.sel(event_time='start_seconds').data)
        stop_seconds = np.array(ds.event_times.sel(event_time='stop_seconds').data)
        names = np.array(ds.event_names.data)
        if 'possible_event_names' in ds.attrs:
            possible_event_names = ds.attrs['possible_event_names']
        elif 'possible_event_names' in ds.event_names.attrs:
            possible_event_names = ds.event_names.attrs['possible_event_names']
        else:
            possible_event_names = []

        out = cls.from_lists(names, start_seconds, stop_seconds, possible_event_names)
        if 'event_categories' in ds:
            cats = {str(cat.event_types.data): str(cat.event_categories.data) for cat in ds.event_categories}
            out = cls(out, categories=cats)
        return out

    def _init_df(self):
        return pd.DataFrame(columns=['name', 'start_seconds', 'stop_seconds'])

    def _append_row(self, df, name, start_seconds, stop_seconds=None):
        if stop_seconds is None:
            stop_seconds = start_seconds
        new_row = pd.DataFrame(np.array([name, start_seconds, stop_seconds])[np.newaxis,:],
                                columns=df.columns)
        return df.append(new_row, ignore_index=True)

    def to_df(self, preserve_empty: bool = True):
        """Convert to pandas.DataFeame

        Args:
            preserve_empty (bool, optional):
                In keeping with the convention that events have identical start and stop times and segments do not,
                empty events are coded with np.nan as both start and stop and
                empty segments are coded as np.nan as start and 0 as stop.
                `from_df()` will obey this convention - if both start and stop are np.nan,
                the name will be a segment,
                if only the start is np.nan (the stop does not matter), the name will be an event
                Defaults to True.

        Returns:
            pandas.DataFrame: with columns name, start_seconds, stop_seconds, one row per event.
        """
        df = self._init_df()
        for name in self.names:
            for start_second, stop_second in zip(self.start_seconds(name), self.stop_seconds(name)):
                df = self._append_row(df, name, start_second, stop_second)
        if preserve_empty:  # ensure we keep events without annotations
            for name, cat in zip(self.names, self.categories.values()):
                if name not in df.name.values:
                    stop_seconds = np.nan if cat == 'event' else 0  # (np.nan, np.nan) -> empty events, (np.nan, some number) -> empty segments
                    df = self._append_row(df, name, start_seconds=np.nan, stop_seconds=stop_seconds)
        # make sure start and stop seconds are numeric
        df['start_seconds'] = pd.to_numeric(df['start_seconds'], errors='coerce')
        df['stop_seconds'] = pd.to_numeric(df['stop_seconds'], errors='coerce')
        return df

    def to_dataset(self):
        df = self.to_df()
        names = df.name.values
        start_seconds = df.start_seconds.values.astype(np.float)
        stop_seconds = df.stop_seconds.values.astype(np.float)

        da_names = xr.DataArray(name='event_names', data=np.array(names, dtype='U128'), dims=['index',])
        da_times = xr.DataArray(name='event_times', data=np.array([start_seconds, stop_seconds]).T, dims=['index','event_time'], coords={'event_time': ['start_seconds', 'stop_seconds']})

        ds = xr.Dataset({da.name: da for da in [da_names, da_times]})
        ds.attrs['time_units'] = 'seconds'
        ds.attrs['possible_event_names'] = self.names  # ensure that we preserve even names w/o events that get lost in to_df
        return ds

    def add_name(self, name, category='segment', times=None, overwrite: bool = False, append: bool = False, sort_after_append: bool = False):
        if times is None:
            times = np.zeros((0,2))

        if name not in self or (name in self and overwrite):
            self.update({name: times})
            self.categories[name] = category
        elif name in self and append:
            self[name] = np.append(self[name], times, axis=0)
            if sort_after_append:
                self[name].sort(axis=0)

    def delete_name(self, name):
        if name in self:
            del self[name]
        if name in self.categories:
            del self.categories[name]

    def add_time(self, name, start_seconds, stop_seconds=None):
        """[summary]

        Args:
            name ([type]): [description]
            start_seconds ([type]): [description]
            stop_seconds ([type], optional): [description]. Defaults to None.
        """
        if stop_seconds is None:
            stop_seconds = start_seconds

        self[name] = np.insert(self[name],
                               len(self[name]),
                               sorted([start_seconds, stop_seconds]),
                               axis=0)

    def move_time(self, name, old_time, new_time):
        """[summary]

        Args:
            name ([type]): [description]
            old_time ([type]): [description]
            new_time ([type]): [description]
        """
        self[name][self[name]==old_time] = new_time

    def delete_time(self, name, time, tol=0):
        nearest_start = self._find_nearest(self.start_seconds(name), time)
        index = np.where(self.start_seconds(name) == nearest_start)[0][0]

        if self.categories[name] == 'segment':
            matching_stop = self.stop_seconds(name)[index]
            event_at_time = matching_stop > time
        elif self.categories[name] == 'event':
            event_at_time = np.abs(time - nearest_start) < tol
        else:
            event_at_time = False

        if event_at_time:
            deleted_time = self[name][index, :]
            self[name] = np.delete(self[name], index, axis=0)
        else:
            deleted_time = None
        return deleted_time

    def select_range(self, name: str, t0: Optional[float] = None, t1: Optional[float] = None, strict: bool = True):
        """Get indices of events within the range.

        Need to start and stop after t0 and before t1 (non-inclusive bounds).

        Args:
            name (str): [description]
            t0 (float, optional): [description]
            t1 (float, optional): [description]
            strict (bool, optional): if true, only matches events that start AND stop within the range,
                           if false, matches events that start OR stop within the range

        Returns:
            List[uint]: List of indices of events within the range
        """

        if t0 is None:
            t0 = 0
        if t1 is None:
            t1 = np.inf

        if strict:
            within_range = np.logical_and(self.start_seconds(name)>t0, self.stop_seconds(name)<t1)
        else:
            starts_in_range = np.logical_and(self.start_seconds(name)>t0, self.start_seconds(name)<t1)
            stops_in_range = np.logical_and(self.stop_seconds(name)>t0, self.stop_seconds(name)<t1)
            within_range = np.logical_or(starts_in_range, stops_in_range)
        within_range_indices = np.where(within_range)[0]
        return within_range_indices

    def filter_range(self, name, t0, t1, strict: bool = False):
        """Returns events within the range.

        Need to start and stop after t0 and before t1 (non-inclusive bounds).

        Args:
            name ([type]): [description]
            t0 ([type]): [description]
            t1 ([type]): [description]
            strict (bool): if true, only matches events that start AND stop within the range,
                           if false, matches events that start OR stop within the range
        Returns:
            List[float]: [N, 2] list of start_seconds and stop_seconds in the range
        """
        indices = self.select_range(name, t0, t1, strict)
        return self[name][indices, :]

    def delete_range(self, name, t0, t1, strict: bool = True):
        """Deletes events within the range.

        Need to start and stop after t0 and before t1 (non-inclusive bounds).

        Args:
            name ([type]): [description]
            t0 ([type]): [description]
            t1 ([type]): [description]
            strict (bool): if true, only matches events that start AND stop within the range,
                           if false, matches events that start OR stop within the range
        Returns:
            int: number of deleted events
        """
        indices = self.select_range(name, t0, t1)
        self[name] = np.delete(self[name], indices, axis=0)
        return len(indices)

    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _infer_categories(self):
        categories = dict()
        for name in self.names:
            if len(self[name])==0:
                if not hasattr(self, 'categories') or name not in self.categories:
                    categories[name] = None
                elif hasattr(self, 'categories') and name in self.categories:
                    categories[name] = self.categories[name]
            else:
                first_start = self.start_seconds(name)[0]
                first_stop = self.stop_seconds(name)[0]

                if (np.isnan(first_start) and np.isnan(first_stop)) or (first_start == first_stop):
                    category = 'event'
                else:
                    category = 'segment'

                categories[name] = category

        return categories

    def _drop_nan(self):
        # remove entries with nan stop or start (but keep their name)
        for name in self.names:
            nan_events = np.logical_or(np.isnan(self.start_seconds(name)),
                                       np.isnan(self.stop_seconds(name)))
            self[name] = self[name][~nan_events]

    @property
    def names(self):
        return list(self.keys())

    def start_seconds(self, key):
        return self[key][:, 0]

    def stop_seconds(self, key):
        return self[key][:, 1]

    def duration_seconds(self, key):
        return self[key][:, 1] - self[key][:, 0]
