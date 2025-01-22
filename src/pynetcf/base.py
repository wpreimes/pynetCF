# Copyright (c) 2023, TU Wien, Department of Geodesy and Geoinformation.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology,
#     Department of Geodesy and Geoinformation nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Base classes for reading and writing time series and images in NetCDF files
using Climate Forecast Metadata Conventions (http://cfconventions.org/).
"""

import os
import time
import numpy as np
import xarray as xr
import datetime
import warnings
from pathlib import Path

from pynetcf.const import alias_a

class DatasetError(Exception):
    pass


class Dataset:
    """
    netCDF file wrapper class that makes some things easier. Built around
    xarray Dataset.

    Parameters
    ----------
    filename : str or Path
        Filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set. if the overwrite
        keyword is set then the file will be overwritten
    name : str, optional (default: None)
        A name for this dataset. Will be written as a global attribute if
        the file is a new file.
    file_format : str, optional
        KEYWORD DEPRECATED in v0.6
        file format
    mode : str, optional (default: "r")
        File access mode. Default is "read-only"
        - "r": means read-only; no data can be modified.
        - "w": means write; a new file is created, an existing file with the
               same name is deleted.
        - "a" and "r+" mean append (in analogy with serial files); an existing
          file is opened for reading and writing.
        Appending s to modes w, r+ or a will enable unbuffered shared access
        to NETCDF3_CLASSIC or NETCDF3_64BIT formatted files. Unbuffered
        access may be useful even if you don"t need shared access, since it
        may be faster for programs that don"t access data sequentially.
        This option is ignored for NETCDF4 and NETCDF4_CLASSIC
        formatted files.
    zlib : boolean, optional (default: True)
        If set, netCDF compression will be used
    complevel : int, optional (default: 4)
        Compression level used from 1 (low compression) to 9 (high compression)
    autoscale : bool, optional (default: True)
        If disabled data will not be automatically scaled when reading and
        writing
    automask : bool, optional (default: True)
        If disabled data will not be masked during reading.
        This means Fill Values will be used instead of NaN.
    """

    def __init__(self,
                 filename,
                 name=None,
                 file_format=None,
                 mode="r",
                 zlib=True,
                 complevel=4,
                 autoscale=True,
                 automask=True):

        if file_format is not None:
            warnings.warn(
                "The `file_format` keyword argument is deprecated "
                "and will be ignored",
                DeprecationWarning, stacklevel=2)

        # DEPRECATED KWS
        self.buf_len = 0 ### ??
        self.file = None
        self.file_format = file_format
        self.autoscale = autoscale
        self.automask = automask

        self.filename = filename
        self._global_attr = dict(
            id=os.path.split(self.filename)[1],
            date_created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        if name is not None:
            self._global_attr["dataset_name"] = name
        self.zlib = zlib
        self.complevel = complevel
        self.mode = mode


        if self.mode in alias_a:
            self.mode = 'a'
        if self.mode == "a" and not os.path.exists(self.filename):
            self.mode = "w"
        if self.mode == "w":
            self._create_file_dir()

        if mode in ['r', 'a']:
            if not os.path.exists(self.filename):
                raise IOError(f"File {self.filename} does not exist.")
            self.dataset = xr.open_dataset(self.filename,
                                           engine="netcdf4")
        else:
            self.dataset = xr.Dataset()

        ## Deprecated
        self.dataset.set_auto_scale(self.autoscale)
        self.dataset.set_auto_mask(self.automask)

    @property
    def global_attr(self):
        return self.dataset.attrs

    def _create_file_dir(self):
        """
        Create directory for file to sit in.
        Avoid race condition if multiple instances are
        writing files into the same directory.
        """
        path = os.path.dirname(self.filename)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                time.sleep(1)
                self._create_file_dir()

    def _set_global_attr(self):
        """
        Write global attributes to NetCDF file.
        """
        self.dataset.attrs.update(self.global_attr)

    def create_dim(self, name, n, axis=0):
        """
        Create dimension for NetCDF file.
        if it does not yet exist

        Parameters
        ----------
        name : str
            Name of the NetCDF dimension.
        n : int
            Size of the dimension.
        axis : int, optional (default: 0)
            The axis along which to create the dimension.
        """
        if name not in self.dataset.dims.keys():
            self.dataset.expand_dims({name: n}, axis=axis)

    def write_var(self,
                  name,
                  data=None,
                  dim=None,
                  attr=None,
                  dtype=None,
                  zlib=None,
                  complevel=None,
                  chunksizes=None,
                  **kwargs):
        """
        Create or overwrite values in a NetCDF variable. The data will be
        written to disk once flush or close is called

        Parameters
        ----------
        name : str
            Name of the NetCDF variable.
        data : np.ndarray, optional
            Array containing the data.
            if not given then the variable will be left empty
        dim : tuple, optional
            A tuple containing the dimension names.
        attr : dict, optional
            A dictionary containing the variable attributes.
            May contain field `_FillValue` to set the fill value for
            this variable manually .
        dtype: str or np.dtype, optional (default: None)
            Data type to use for encoding this variable.
            If not given data.dtype will be used
        zlib: bool, optional (default: None)
            Can be used to force activate/deactivate compression for this
            variable. If not specified, use the chosen global attribute
            (self.zlib).
        complevel: int, optional (default: 4)
            Explicit compression level for this variable
            if not given then global attribute is used
        chunksizes : tuple, optional
            chunksizes can be used to manually specify the
            HDF5 chunksizes for each dimension of the variable.
        **kwargs:
            Additional kwargs are forwarded to xr.DataArray.
        """

        attr = attr or dict()

        if "_FillValue" in attr:
            fill_value = attr.pop("_FillValue")
        else:
            fill_value = None

        if dtype is None:
            dtype = data.dtype

        if zlib is None:
            zlib = self.zlib

        if not np.issubdtype(dtype, np.number):
            # Only numeric data can be compressed
            zlib = False

        if complevel is None:
            complevel = self.complevel

        if name in self.dataset.data_vars:
            var = self.dataset[name]
        else:
            var = xr.DataArray(name=name, data=data, dims=dim, **kwargs)
            var.encoding["_FillValue"] = fill_value
            var.encoding["dtype"] = dtype
            var.encoding["zlib"] = zlib
            var.encoding["complevel"] = complevel

            if chunksizes is not None:
                var = var.chunk(chunksizes)

            var = self.dataset.createVariable(name,
                                              dtype,
                                              dim,
                                              fill_value=fill_value,
                                              zlib=zlib,
                                              complevel=complevel,
                                              chunksizes=chunksizes,
                                              **kwargs)

        for attr_name in attr:
            attr_value = attr[attr_name]
            var.setncattr(attr_name, attr_value)

        var.set_auto_scale(self.autoscale)
        if data is not None:
            var[:] = data

    def append_var(self, name, data, **kwargs):
        """
        append data along unlimited dimension(s) of variable

        Parameters
        ----------
        name : string
            Name of variable to append to.
        data : numpy.array
            Numpy array of correct dimension.

        Raises
        ------
        IOError
            if appending to variable without unlimited dimension
        """
        if name in self.dataset.variables.keys():
            var = self.dataset.variables[name]
            dim_unlimited = []
            key = []
            for index, dim in enumerate(var.dimensions):
                unlimited = self.dataset.dimensions[dim].isunlimited()
                dim_unlimited.append(unlimited)
                if not unlimited:
                    # if the dimension is not unlimited set the slice to :
                    key.append(slice(None, None, None))
                else:
                    # if unlimited set slice of this dimension to
                    # append meaning
                    # [var.shape[index]:]
                    key.append(slice(var.shape[index], None, None))

            dim_unlimited = np.array(dim_unlimited)
            nr_unlimited = np.where(dim_unlimited)[0].size
            key = tuple(key)
            # if there are unlimited dimensions we can do an append
            if nr_unlimited > 0:
                var[key] = data
            else:
                raise IOError(
                    "Cannot append to variable that has no unlimited dimension")
        else:
            self.write_var(name, data, **kwargs)

    def read_var(self, name):
        """
        reads variable from netCDF file

        Parameters
        ----------
        name : string
            name of the variable
        """

        if self.mode in ["r", "r+", "a"]:
            if name in self.dataset.variables.keys():
                return self.dataset.variables[name][:]

    def add_global_attr(self, name, value):
        """
        Add global attribute.

        Parameters
        ----------
        name : str
            Name.
        value : str or number
            Value.
        """
        self.global_attr[name] = value

    def flush(self):
        """
        Flush data to disk.
        """
        if self.dataset is not None:
            if self.mode in ["w", "r+", "a"]:
                self._set_global_attr()
                self.dataset.sync()

    def close(self):
        """
        Flush and close file.
        """
        if self.dataset is not None:
            self.flush()
            self.dataset.close()
            self.dataset = None

    def __enter__(self):
        """
        ContextManager enter.
        """
        return self

    def __exit__(self, value_type, value, traceback):
        """
        ContextManager exit.
        """
        self.close()

    def __del__(self):
        """
        Destructor.
        """
        if hasattr(self, "dataset"):
            self.close()
