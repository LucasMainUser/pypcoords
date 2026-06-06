from typing import (
    Any, 
    Protocol,
    SupportsIndex,
    TypeAlias,
    Mapping,
    Hashable,
    Iterable,
    Self,
    Sized,
    Optional,
    Sequence,
    runtime_checkable
)
from dataclasses import dataclass
from operator import index as to_index

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from matplotlib.pyplot import get_cmap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    Normalize, 
    Colormap, 
    ListedColormap,
    LinearSegmentedColormap,
    to_rgba,
    is_color_like
)
from matplotlib.path import Path
from matplotlib.patches import PathPatch


@runtime_checkable
class DataframeLike(Protocol):
    columns: Iterable[Hashable]
    def __getitem__(self, key: Hashable, /) -> npt.ArrayLike: ...

@runtime_checkable
class Transformer(Protocol):
    def fit(self, values: npt.ArrayLike, /) -> Self: ...
    def transform(self, values: npt.ArrayLike, /) -> npt.NDArray: ...
    def inverse_transform(self, values: npt.ArrayLike, /) -> npt.NDArray: ...

@dataclass(slots=True)
class NumberTransformer:
    dtype: Optional[type] = None
    digits: Optional[int] = None

    def fit(self, values: npt.ArrayLike, /, digits: Optional[int] = None) -> Self:
        self.dtype = np.asarray(values).dtype
        self.digits = digits
        return self

    def transform(self, values: npt.ArrayLike, /) -> npt.NDArray:
        array = np.asarray(values, dtype=self.dtype, copy=True)
        if self.digits is not None:
            array = np.round(array, decimals=self.digits)
        return array

    def inverse_transform(self, values: npt.ArrayLike, /) -> npt.NDArray:
        array = np.asarray(values)
        if self.digits is not None:
            array = np.round(array, decimals=self.digits)

        if np.issubdtype(self.dtype, np.integer):
            return np.round(array).astype(self.dtype)

        return array.astype(self.dtype, copy=True)

@dataclass(slots=True)
class CategoricalTransformer:
    categories: Optional[npt.NDArray] = None

    def fit(self, values: npt.ArrayLike, /) -> Self:
        array = np.asarray(values)
        self.categories = np.unique(array)
        return self
    
    def transform(self, values: npt.ArrayLike, /) -> npt.NDArray:
        array = np.asarray(values)
        return np.searchsorted(self.categories, array)
    
    def inverse_transform(self, values: npt.ArrayLike, /) -> npt.NDArray:
        array = np.asarray(values, dtype=int)
        return self.categories[array]
    
    @property
    def num_categories(self) -> int:
        if self.categories is None:
            return 0
        return len(self.categories)
 
@dataclass(slots=True)
class MinMaxTransformer: 
    lower: float
    upper: float

    vmin: Optional[float]=None
    vmax: Optional[float]=None
    
    def fit(self, values: npt.ArrayLike, /) -> Self:
        array = np.asarray(values, dtype=float)
        self.vmin = np.min(array)
        self.vmax = np.max(array)
        return self
    
    def transform(self, values: npt.ArrayLike, /) -> npt.NDArray:
        array = np.asarray(values, dtype=float)

        if self.vmax == self.vmin:
            midpoint = (self.lower + self.upper) / 2
            return np.full(array.shape, midpoint, dtype=float)

        scale = (self.upper - self.lower) / (self.vmax - self.vmin)
        return (array - self.vmin) * scale + self.lower

    def inverse_transform(self, values: npt.ArrayLike, /) -> npt.NDArray:
        array = np.asarray(values, dtype=float)

        if self.vmax == self.vmin:
            return np.full(array.shape, self.vmin, dtype=float)

        scale = (self.vmax - self.vmin) / (self.upper - self.lower)
        return (array - self.lower) * scale + self.vmin

Category:       TypeAlias = str
HexColor:       TypeAlias = str
ColumnName:     TypeAlias = Hashable
RGB:            TypeAlias = tuple[float, float, float]
RGBA:           TypeAlias = tuple[float, float, float, float]
ColorLike:      TypeAlias = HexColor | RGB | RGBA
Ticks:          TypeAlias = Sequence[float]
TickLabels:     TypeAlias = Sequence[str]
Vertices:       TypeAlias = Sequence[float]
Codes:          TypeAlias = Sequence[float]
ColumnsMapping: TypeAlias = Mapping[Hashable, npt.NDArray]
RowStyle:       TypeAlias = Mapping[str, Any]
ColumnValues:   TypeAlias = Sequence[Any]
RowValues:      TypeAlias = Sequence[Any]


def type_name(it: Any, /) -> str:
    return getattr(type(it), '__name__')

def move_to_end(data: list[Any], value: Any, /) -> list[Any]:
    data.remove(value)
    data.append(value)
    return data

def iter_rows(columns: Iterable[ColumnValues], /) -> Iterable[RowValues]:
    return zip(*columns, strict=True)

def dict_from_lists(keys: Iterable[Hashable], values: Iterable[Any], /) -> dict[Hashable, Any]:
    return dict(zip(keys, values, strict=True))

def generate_indices(stop: int, /) -> list[int]:
    return list(range(stop))

def assert_same_lengths(*data: Sized) -> None:
    it = iter(data)
    try:
        expected = len(next(it))
    except StopIteration:
        return  

    for obj in it:
        if len(obj) == expected:
            continue
        raise ValueError('All inputs must have the same length')

def vector(data: npt.ArrayLike, /, copy: bool=True) -> npt.NDArray:
    array = np.atleast_1d(data).ravel()
    if copy:
        return array.copy()
    return array

def infer_transformer(values: npt.ArrayLike, /) -> Transformer:
    array = np.asarray(values)

    if np.issubdtype(array.dtype, np.number):
        return NumberTransformer()
    
    try:
        np.astype(array, dtype=float)
        return NumberTransformer()
    except (TypeError, ValueError):
        return CategoricalTransformer()

def sort_keys(m: Mapping[Hashable, Any], /, reverse: bool=False) -> dict[Hashable, Any]:
    return {key: m[key] for key in sorted(m.keys(), reverse=reverse) }

def infer_columns(data: Mapping | DataframeLike | npt.ArrayLike, /) -> list[Hashable]:
    if isinstance(data, DataframeLike):
        return list(data.columns)
    if isinstance(data, Mapping):
        return list(data.keys())
    try:
        matrix = np.asarray(data)

        if matrix.ndim != 2:
            raise ValueError('Expected 2D array-like')
        num_columns = matrix.shape[1]
        return list(range(num_columns))
    except Exception as error:
        raise ValueError(f'Could not infer columns from data. Details: {error}') from error

def map_columns(data: npt.ArrayLike | DataframeLike | Mapping[Hashable, npt.ArrayLike], columns: Iterable[Hashable], /) -> ColumnsMapping:
    if isinstance(
        data, (Mapping, DataframeLike)
    ):
        return {column: vector(data[column]) for column in columns}
    
    matrix = np.asarray(data)

    if matrix.ndim != 2:
        raise ValueError('Expected 2D array-like')
    return {column: vector(matrix[:, column]) for column in columns}
     
def map_numeric_transformers(mapping: ColumnsMapping, /, decimals: Mapping[Hashable, int] | None=None) -> dict[Hashable, Transformer]:
    if decimals is None:
        decimals = {}
    
    def fit_transformer(column: Hashable, values: Sequence, /) -> Transformer:
        transformer = infer_transformer(values)
        if isinstance(transformer, NumberTransformer):
            return transformer.fit(values, decimals.get(column))
        return transformer.fit(values)
    
    return {
        column: fit_transformer(column, values) 
        for column, values in mapping.items()
    }

def map_reescalers(mapping: ColumnsMapping, /, lower: float, upper: float) -> dict[Hashable, Transformer]:
    return {
        column: MinMaxTransformer(lower, upper).fit(values) 
        for column, values in mapping.items()
    }

def map_parallel_axes(
        columns: Iterable[ColumnName], 
        /,   
        cmap: Optional[str | Colormap]=None,
        norm: Optional[Normalize]=None,
        ax: Optional[plt.Axes]=None,
        colorbar_at_end: bool=False,
    ) -> dict[ColumnName, plt.Axes]:

    columns = list(columns)
    total = len(columns)

    ax = resolve_axes(ax)
    
    if not colorbar_at_end:
        parallel_axes = [ax] + [ ax.twinx() for _ in range(total - 1) ]
        return dict_from_lists(columns, parallel_axes)
    
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = plt.colorbar(scalar_mappable, ax=ax, pad=0.0)

    parallel_axes = [ax] + [ ax.twinx() for _ in range(total - 2) ] + [colorbar.ax]
    return dict_from_lists(columns, parallel_axes)

def map_column_labels(columns: Iterable[ColumnName], labels: Optional[Iterable[str] | Mapping[ColumnName, str]]=None, /) -> dict[ColumnName, str]:
    columns = list(columns)

    if labels is None:
        return dict_from_lists(columns, columns)

    if isinstance(labels, Mapping):
        missing = set(columns).difference(labels.keys() )
        if missing:
            missing = sorted(missing)
            raise ValueError(f'Missing labels for columns: {missing}')
        return dict(labels)
        
    return dict_from_lists(columns, labels)

def cubic_bezier_path(x: npt.ArrayLike, y: npt.ArrayLike, /) -> tuple[Vertices, Codes]:
    x_array = vector(x)
    y_array = vector(y)

    assert_same_lengths(x_array, y_array)

    codes: Codes = []
    vertices: Vertices = []
    
    is_first_point = True
    num_segments = len(x) - 1

    for index in range(num_segments):
        x_start = x_array[index]
        y_start = y_array[index]

        x_final = x_array[index + 1]
        y_final = y_array[index + 1]

        delta_x = x_final - x_start

        start_point     = (x_start, y_start)
        control_point_1 = (x_start + 1/3 * delta_x, y_start)
        control_point_2 = (x_start + 2/3 * delta_x, y_final)
        final_point     = (x_final, y_final)

        if is_first_point:
            is_first_point = False
            vertices.append(start_point)
            codes.append(Path.MOVETO)
        
        vertices.extend([control_point_1, control_point_2, final_point])
        codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

    return vertices, codes

def resolve_axes(ax: Optional[plt.Axes | None]=None, /) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, plt.Axes):
        raise TypeError(
            f'Invalid type for <ax>. Expected matplotlib.axes.Axes, got {type_name(ax)} instead.'
        )
    return ax

def transform_rgba(data: str | RGB | RGBA, /, alpha: Optional[float]=None) -> RGBA:
    if (
        isinstance(data, tuple) 
        and len(data) in (3,4) 
        and max(data) > 1
    ):
        data = tuple(value / 255 for value in data)
    return to_rgba(data, alpha=alpha)

def resolve_color_system(
        to_color_array: npt.ArrayLike, 
        /,
        palette: Optional[ColorLike | Mapping[float, ColorLike] | Colormap] = None,
        colormap_gradient: Optional[int]=None,
        vmin: Optional[float]=None,
        vmax: Optional[float]=None 
    ) -> tuple[Colormap, Normalize, np.ndarray[RGBA, None] ]:

    if colormap_gradient is None:
        colormap_gradient = 256

    values = np.asarray(to_color_array)
    
    if vmin is None:
        vmin = np.min(values)
    
    if vmax is None:
        vmax = np.max(values)

    if isinstance(palette, Mapping):    
        palette = dict(palette)
        segments = [
            (value, color) for value, color in sort_keys(palette, reverse=False).items()
        ]
        
        cmap = LinearSegmentedColormap.from_list('linear_segment_colormap', segments, N=colormap_gradient)
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors_array = np.array([
            transform_rgba(palette[value]) for value in values
        ])
        return cmap, norm, colors_array
    
    cmap = None

    if is_color_like(palette):
        colors = [transform_rgba(palette)]
        cmap = ListedColormap(colors)
        
    if cmap is None:
        cmap = get_cmap(palette)
        
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors_array = cmap(norm(values))
    
    return cmap, norm, colors_array
    


def parallel_plot(
        data: Mapping[ColumnName, npt.ArrayLike] | DataframeLike | npt.ArrayLike, 
        columns: Iterable[ColumnName] | None=None,
        column_labels: Iterable[str] | Mapping[ColumnName, str] | None=None,
        color_column: Optional[ColumnName]=None,
        palette: Optional[ColorLike | Mapping[Category, ColorLike] | Colormap]=None,
        colormap_gradient: Optional[int]=None,
        linewidth: Optional[float]=None,
        alpha: Optional[float]=None,
        highlight_rows: Mapping[SupportsIndex, RowStyle] | None=None,
        decimals: Mapping[ColumnName, int] | None=None,
        ax: Optional[plt.Axes]=None,
        show_colorbar: bool=True
    ) -> plt.Axes:
    '''
    Create a parallel coordinates plot from tabular data.

    Each row is drawn as a smooth curve across vertical axes (one per column).
    All columns are scaled to [0, 1] so different types and ranges can be compared.

    Parameters
    ----------
    data : Mapping, DataFrame-like, or 2D array
        Input data. Supported formats:
        - dict-like: {column_name: 1D array}
        - DataFrame: pandas, polars, etc.
        - 2D array: shape (n_rows, n_columns)
        All columns must have the same length.

    columns : iterable, optional
        Columns to include. Defaults to all.

    column_labels : iterable of str or mapping, optional
        Labels for the x-axis.
        - If iterable: must match order of `columns`
        - If mapping: {column_name: label}
        - If None, column names are used.

    color_column : column name, optional
        Column used for coloring lines. Defaults to last column.

    palette : color, mapping, colormap-name or colormap, optional
        - colormap → continuous colors
        - dict → category → color
        - single color → all lines same color

    colormap_gradient : int, optional
        Number of color steps used in the gradient (must be > 0).

        Higher values → smoother, more continuous color transitions.  
        Lower values → fewer colors, more visible steps (banded look).

    linewidth : float, optional
        Line width (default 1.0).

    alpha : float, optional
        Line transparency (default 0.7).

    highlight_rows : dict, optional
        Custom styles for specific rows: {row_index: style_dict}.

    decimals : dict, optional
        Decimal formatting per column.

    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates one if None.

    show_colorbar : bool, default True
        Show colorbar for the color column.

    Returns
    -------
    matplotlib.axes.Axes
    
    Raises
    ------
    ValueError
        - Fewer than 2 columns → need at least two axes.
        - `color_column` not in `columns` → invalid color reference.
        - Columns have different lengths → inconsistent row count.
        - Invalid tabular input → unsupported format or shape mismatch.
        - Incomplete categorical `palette` → missing category colors.
        - `columns` and `column_labels` size mismatch → labels must align.
        - No valid values after processing → cannot scale or plot.

    Notes
    -----
    - Columns are scaled independently to [0, 1]
    - Categorical data is encoded automatically
    - Curves are cubic Bézier
    - Color axis moves to the end if colorbar is shown

    Examples
    --------
    >>> parallel_plot(df)

    >>> parallel_plot(df, color_column="C", alpha=0.5)
    '''
    if alpha is None:
        alpha = 0.7

    if linewidth is None:
        linewidth = 1.0
    
    if highlight_rows is None:
        highlight_rows = {}
    
    # NOTE: These variables defines: 
    # 1) y-axis limits (bottom, top) 
    # 2) colorbar.norm limits
    ymin, ymax = -0.05, 1.05
    
    # NOTE: Resolve column subsets
    if columns is None:
        columns = infer_columns(data)
    columns = list(columns)

    num_columns = len(columns)

    if num_columns < 2:
        raise ValueError('parallel-plot requires at least two columns')

    # NOTE: Resolve column to color
    if color_column is None:
        color_column = columns[-1]
    
    if color_column not in columns:
        raise ValueError(f'color_column ({color_column!r}) not in {columns!r}')
    
    # NOTE: Resolve x-labels
    labels = map_column_labels(columns, column_labels)
    
    # NOTE: Transform data into a columnar mapping
    # Assert all arrays have same length
    series = map_columns(data, columns)
    assert_same_lengths(*series.values())

    # NOTE: Fit a numeric-transformer for each column
    # Transform non-numeric to number
    transformers = map_numeric_transformers(series, decimals)
    series_transformed = {
        column: transformers[column].transform(values) 
        for column, values in series.items()
    }

    # NOTE: Fit a scaler for each numeric column
    # Transform numbers to 0-1 fractions
    scalers = map_reescalers(series_transformed, lower=0, upper=1)
    series_reescaled = {
        column: scalers[column].transform(values) 
        for column, values in series_transformed.items()
    }   

    # NOTE: Resolve colors
    # Normalize pallete if is mapping
    if isinstance(palette, Mapping):
        color_column_transformer = transformers[color_column]
        color_column_scaler = scalers[color_column]
        color_column_values = series[color_column]
        
        palette = dict(palette)

        missing_categories = set(color_column_values).difference(palette)
        if missing_categories:
            missing_categories = sorted(missing_categories)
            raise ValueError(
                f'Invalid palette for color_column={color_column!r}. '
                f'The following categories are missing from the palette: {missing_categories}. '
            )
        
        palette = {
            color_column_transformer.transform(category): color 
            for category, color in palette.items() 
        }
        palette = {
            color_column_scaler.transform(index): color
            for index, color in palette.items()
        }
        
    cmap, norm, colors_array = resolve_color_system(
        series_reescaled[color_column], 
        palette=palette, 
        colormap_gradient=colormap_gradient,
        vmin=ymin,
        vmax=ymax)
    
    # NOTE: Create parallel axes
    # If show colorbar, then color-column should be moved to the end
    ax = resolve_axes(ax)

    if show_colorbar:
        columns = move_to_end(columns, color_column)
    parallel_axes = map_parallel_axes(columns, cmap=cmap, norm=norm, ax=ax, colorbar_at_end=show_colorbar)
    
    # NOTE: Configure vertical-axes (position, ticks, ticklabels)
    for index, column_name in enumerate(parallel_axes):
        axes = parallel_axes[column_name]
        scaler = scalers[column_name]
        transformer = transformers[column_name]

        axes.set_ylim(ymin, ymax)
        
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)

        if axes is not ax:
            axes.spines['left'].set_visible(False)
            axes.yaxis.set_ticks_position('right')

            axes_position = index / (num_columns - 1)
            axes.spines['right'].set_position(('axes', axes_position))

        if isinstance(transformer, CategoricalTransformer):
            indices = np.arange(transformer.num_categories)
            yticks = scaler.transform(indices)
            yticklabels = transformer.inverse_transform(indices)
        else:
            yticks = np.linspace(0, 1, 5, endpoint=True)
            yticklabels = scaler.inverse_transform(yticks)
            yticklabels = transformer.inverse_transform(yticklabels)

        unique_yticklabels = np.unique(yticklabels)

        if len(unique_yticklabels) == 1:
            yticks = [0.5]
            yticklabels = unique_yticklabels
            
        axes.set_yticks(yticks, yticklabels)

    # NOTE: Configure horizontal axes
    x_positions = generate_indices(num_columns)

    ax.set_xticks(ticks=x_positions, labels=[labels[column_name] for column_name in columns])
    ax.tick_params(axis='x', which='major', pad=7, zorder=10)

    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_top()

    # NOTE: Resolve highlight rows
    highlight_rows = {
        to_index(row): style for row, style in highlight_rows.items()
    }

    # NOTE: Draw Bezier curvers
    rows = iter_rows(series_reescaled[column_name] for column_name in columns)

    for row_index, row_values in enumerate(rows):
        row_style = highlight_rows.get(row_index, None)

        if row_style is None:
            row_style = {
                'edgecolor': colors_array[row_index],
                'linewidth': linewidth,
                'alpha': alpha
            }
        row_style = dict(row_style)
        row_style.setdefault('facecolor', 'none')
        
        vertices, codes = cubic_bezier_path(x_positions, row_values)
        path = Path(vertices, codes)

        patch = PathPatch(path, **row_style)
        ax.add_patch(patch)

    return ax

