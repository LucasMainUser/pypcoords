# pypcoords

Simple and flexible **parallel coordinates plot** implementation built on top of `matplotlib`.

Supports multiple data formats out of the box, including **NumPy**, **pandas**, and **polars**.

---

## ✨ Features

* 📊 Parallel coordinates visualization
* 🔌 Works with:

  * dict / mappings
  * NumPy arrays
  * pandas DataFrame
  * polars DataFrame
* 🎨 Flexible color system (colormap or custom colors)
* 🧠 Automatic handling of:

  * numeric data
  * categorical data
* 🧵 Smooth curves using cubic Bézier interpolation
* 🎯 Row highlighting support

---

## 📦 Installation

```bash
pip install pypcoords
```

---

## 🚀 Quick Example

```python
from pypcoords import parallel_plot

data = {
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9],
}

parallel_plot(data)
```

---

## 🎨 With colors

```python
parallel_plot(
    data,
    colormap="viridis",
    colormap_column="C"
)
```

---

## 🔍 Highlight specific rows

```python
parallel_plot(
    data,
    highlight_rows={
        1: {"edgecolor": "red", "linewidth": 2.5}
    }
)
```

---

## 📚 Supported input formats

* Mapping of column names to arrays
* pandas DataFrame
* polars DataFrame
* 2D array-like (NumPy, lists, etc.)

---

## 📄 License

MIT

---

## 🤝 Contributing

Contributions are welcome!
