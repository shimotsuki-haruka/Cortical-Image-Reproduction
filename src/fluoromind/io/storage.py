"""Storage strategies for saving and loading data in various formats."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Dict, Any, Type, List, Optional, TYPE_CHECKING
import numpy as np
import json
from pathlib import Path
from importlib.util import find_spec

# Type definitions for type checking
if TYPE_CHECKING:
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

# Check for optional dependencies
HAS_H5PY = find_spec("h5py") is not None
HAS_PYARROW = find_spec("pyarrow") is not None

# Import optional dependencies if available
if HAS_H5PY:
    import h5py  # type: ignore
else:
    h5py = None  # type: ignore

if HAS_PYARROW:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
else:
    pa = None  # type: ignore
    pq = None  # type: ignore


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dict."""

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable": ...


class SerializationStrategy(ABC):
    """Abstract base class for serialization strategies."""

    @abstractmethod
    def save(self, obj: Dict[str, Any], filepath: str) -> None:
        """Save object to file."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load object from file."""
        pass


class JSONStrategy(SerializationStrategy):
    """JSON serialization strategy."""

    def save(self, obj: Dict[str, Any], filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(obj, f, cls=NumpyJSONEncoder)

    def load(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r") as f:
            return json.load(f)


class HDF5Strategy(SerializationStrategy):
    """HDF5 serialization strategy for large numerical data."""

    def _save_dict_to_group(self, h5group: "h5py.Group", d: Dict[str, Any]) -> None:
        for key, item in d.items():
            if not isinstance(key, str):
                raise TypeError(f"Dictionary keys must be strings, got {type(key)}")
            try:
                if isinstance(item, dict):
                    group = h5group.create_group(key)
                    self._save_dict_to_group(group, item)
                else:
                    h5group.create_dataset(key, data=item)
            except Exception as err:
                raise ValueError(f"Failed to save item with key '{key}': {str(err)}") from err

    def _load_dict_from_group(self, h5group):
        """Recursively load dictionary from HDF5 group."""
        d = {}
        for key in h5group.keys():
            item = h5group[key]
            if isinstance(item, h5py.Group):
                d[key] = self._load_dict_from_group(item)
            else:
                d[key] = item[()]
        return d

    def save(self, obj: Dict[str, Any], filepath: str) -> None:
        with h5py.File(filepath, "w") as f:
            self._save_dict_to_group(f, obj)

    def load(self, filepath: str) -> Dict[str, Any]:
        with h5py.File(filepath, "r") as f:
            return self._load_dict_from_group(f)


class NPZStrategy(SerializationStrategy):
    """NumPy's NPZ serialization strategy for numerical arrays."""

    def save(self, obj: Dict[str, Any], filepath: str) -> None:
        # Convert non-array values to arrays for storage
        np_dict = {}
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                np_dict[key] = value
            else:
                # Store metadata as arrays
                np_dict[f"{key}_metadata"] = np.array([str(value)])
        np.savez_compressed(filepath, **np_dict)

    def load(self, filepath: str) -> Dict[str, Any]:
        with np.load(filepath) as data:
            result = {}
            for key in data.files:
                if key.endswith("_metadata"):
                    # Restore original metadata
                    base_key = key.replace("_metadata", "")
                    result[base_key] = data[key][0]
                else:
                    result[key] = data[key]
            return result


class ParquetStrategy(SerializationStrategy):
    """Apache Parquet serialization strategy for tabular numerical data."""

    def _convert_to_table(self, obj: Dict[str, Any]) -> "pa.Table":
        """Convert dictionary to Arrow table."""
        arrays = []
        names = []
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    arrays.append(pa.array(value))
                    names.append(key)
                else:
                    # Handle multi-dimensional arrays by flattening
                    flat_array = value.reshape(value.shape[0], -1)
                    for i in range(flat_array.shape[1]):
                        arrays.append(pa.array(flat_array[:, i]))
                        names.append(f"{key}_dim_{i}")
            else:
                # Handle scalar values
                arrays.append(pa.array([value]))
                names.append(key)
        return pa.Table.from_arrays(arrays, names=names)

    def save(self, obj: Dict[str, Any], filepath: str) -> None:
        table = self._convert_to_table(obj)
        pq.write_table(table, filepath)

    def load(self, filepath: str) -> Dict[str, Any]:
        table = pq.read_table(filepath)
        result = {}

        # Group columns by prefix to reconstruct arrays
        column_groups = {}
        for col in table.column_names:
            if "_dim_" in col:
                base_name = col.split("_dim_")[0]
                if base_name not in column_groups:
                    column_groups[base_name] = []
                column_groups[base_name].append(col)
            else:
                # Handle scalar or 1D array data
                result[col] = table[col].to_numpy()
                if len(result[col]) == 1:
                    result[col] = result[col][0]

        # Reconstruct multi-dimensional arrays
        for base_name, columns in column_groups.items():
            columns.sort(key=lambda x: int(x.split("_dim_")[1]))
            arrays = [table[col].to_numpy() for col in columns]
            result[base_name] = np.column_stack(arrays)

        return result


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and other special types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"_type": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype), "shape": obj.shape}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class BaseResult(Serializable):
    """Base class for analysis results that can be serialized.

    This class implements the Serializable protocol and provides basic
    serialization functionality for numpy arrays and other common data types.
    All result container classes should inherit from this class.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result object to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all attributes of the result object.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = {
                    "_type": "ndarray",
                    "data": value.tolist(),
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                }
            elif isinstance(value, Serializable):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseResult":
        """Create a result object from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the serialized result data.

        Returns
        -------
        BaseResult
            New instance of the result class.
        """
        instance = cls()
        for key, value in data.items():
            if isinstance(value, dict) and value.get("_type") == "ndarray":
                setattr(instance, key, np.array(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"]))
            else:
                setattr(instance, key, value)
        return instance


class StrategyRegistry:
    """Registry for available serialization strategies."""

    _strategies: Dict[str, Type[SerializationStrategy]] = {}
    _extensions: Dict[str, Type[SerializationStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: Type[SerializationStrategy], extension: str) -> None:
        """Register a strategy if its dependencies are available."""
        if not issubclass(strategy_class, SerializationStrategy):
            raise TypeError(f"Strategy must inherit from SerializationStrategy: {strategy_class}")
        if not extension.startswith("."):
            raise ValueError(f"Extension must start with '.': {extension}")

        cls._strategies[strategy_class.__name__] = strategy_class
        cls._extensions[extension] = strategy_class

    @classmethod
    def get_strategy(cls, extension: str) -> Type[SerializationStrategy]:
        """Get appropriate strategy for file extension."""
        if not extension.startswith("."):
            extension = f".{extension}"
        strategy = cls._extensions.get(extension)
        if strategy is None:
            import warnings

            warnings.warn(f"No strategy found for extension {extension}, falling back to JSON", stacklevel=2)
            return JSONStrategy
        return strategy

    @classmethod
    def initialize(cls) -> None:
        """Initialize available strategies based on installed packages."""
        # Always register JSON (built-in)
        cls.register(JSONStrategy, ".json")

        # Register HDF5 if available
        if HAS_H5PY:
            cls.register(HDF5Strategy, ".h5")

        # Register NPZ (requires numpy, which is a core dependency)
        cls.register(NPZStrategy, ".npz")

        # Register Parquet if available
        if HAS_PYARROW:
            cls.register(ParquetStrategy, ".parquet")


# Initialize registry at module load
StrategyRegistry.initialize()


class SaveableMixin:
    """Mixin class providing save/load functionality with strategy pattern.

    Parameters
    ----------
    serializer : SerializationStrategy, optional
        The serialization strategy to use, by default HDF5Strategy if available,
        otherwise JSONStrategy
    state_attrs : List[str], optional
        List of attributes to include in serialization
    """

    def __init__(self, serializer: Optional[SerializationStrategy] = None, state_attrs: Optional[List[str]] = None):
        self._serializer: SerializationStrategy = serializer or (HDF5Strategy() if HAS_H5PY else JSONStrategy())
        self._state_attrs: List[str] = state_attrs or []

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"serializer": self._serializer, "state_attrs": self._state_attrs}

    def set_params(self, **params) -> "SaveableMixin":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : SaveableMixin
            Estimator instance.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
            setattr(self, f"_{key}", value)
        return self

    def set_serializer(self, serializer: SerializationStrategy) -> None:
        """Change the serialization strategy.

        Parameters
        ----------
        serializer : SerializationStrategy
            The new serialization strategy to use
        """
        if not isinstance(serializer, SerializationStrategy):
            raise TypeError("serializer must be an instance of SerializationStrategy")
        self._serializer = serializer

    def set_state_attrs(self, attrs: List[str]) -> None:
        """Set the attributes to be included in serialization.

        Parameters
        ----------
        attrs : List[str]
            List of attribute names to include in serialization
        """
        if not isinstance(attrs, list) or not all(isinstance(attr, str) for attr in attrs):
            raise TypeError("attrs must be a list of strings")
        self._state_attrs = attrs

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the object.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the current state

        Raises
        ------
        ValueError
            If no state attributes are defined
        AttributeError
            If a required state attribute is missing
        """
        if not self._state_attrs:
            raise ValueError("No state attributes defined. Call set_state_attrs first.")

        state = {}
        for attr in self._state_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required state attribute: {attr}")
            value = getattr(self, attr)
            state[attr] = value.to_dict() if isinstance(value, Serializable) else value
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the object.

        Parameters
        ----------
        state : Dict[str, Any]
            Dictionary containing the state to set
        """
        for attr in self._state_attrs:
            if attr in state:
                if hasattr(self, attr):
                    current_value = getattr(self, attr)
                    if isinstance(current_value, Serializable):
                        setattr(self, attr, current_value.__class__.from_dict(state[attr]))
                    else:
                        setattr(self, attr, state[attr])

    def save(self, filepath: str | Path) -> None:
        """Save the current state.

        Parameters
        ----------
        filepath : str | Path
            Path to save the state to
        """
        filepath = Path(filepath)
        state = self.get_state()

        strategy_class = StrategyRegistry.get_strategy(filepath.suffix)
        self._serializer = strategy_class()
        self._serializer.save(state, str(filepath))

    @classmethod
    def load(cls, filepath: str | Path) -> "SaveableMixin":
        """Load state from file.

        Parameters
        ----------
        filepath : str | Path
            Path to load the state from

        Returns
        -------
        SaveableMixin
            New instance with loaded state
        """
        filepath = Path(filepath)
        instance = cls()

        strategy_class = StrategyRegistry.get_strategy(filepath.suffix)
        instance._serializer = strategy_class()

        state = instance._serializer.load(str(filepath))
        instance.set_state(state)
        return instance
