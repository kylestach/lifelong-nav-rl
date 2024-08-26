from dataclasses import fields
from flax.struct import dataclass as _flax_dataclass, field as _flax_field
from typing import dataclass_transform


def dataclass_from_dict(_dataclass, kwarg_dict):
    field_names = {f.name for f in fields(_dataclass) if f.init}
    filtered_kwarg_dict = {k: v for k, v in kwarg_dict.items() if k in field_names}
    return _dataclass(**filtered_kwarg_dict)


def dataclass_from_dataclass(_dataclass, dataclass_instance, defaults=None):
    field_names = {f.name for f in fields(_dataclass) if f.init}
    filtered_kwarg_dict = {
        k: v for k, v in dataclass_instance.__dict__.items() if k in field_names
    }

    if defaults is None:
        defaults_kwarg_dict = {}
    elif isinstance(defaults, dict):
        defaults_kwarg_dict = defaults
    else:
        defaults_kwarg_dict = defaults.__dict__

    return _dataclass(**{**defaults_kwarg_dict, **filtered_kwarg_dict})


@dataclass_transform(field_specifiers=(_flax_field,))  # type: ignore[literal-required]
def dataclass(cls):
    def _to(self, other_cls):
        # If `other` is a dataclass _instance_ (with a different type), use it as defaults and convert with its class
        if isinstance(other_cls, type):
            return dataclass_from_dataclass(other_cls, self)
        else:
            return dataclass_from_dataclass(type(other_cls), self, other_cls)

    cls = _flax_dataclass(cls)
    cls.to = _to

    return cls


if __name__ == "__main__":

    @dataclass
    class A:
        a: int
        b: int

    @dataclass
    class B:
        a: int
        b: int
        c: int
    
    @dataclass
    class C:
        a: int
        b: int
        c: int
        d: int

    print(dataclass_from_dict(A, {"a": 1, "b": 2, "c": 3}))
    b_instance = B(1, 2, 3)
    print(dataclass_from_dataclass(A, b_instance))

    print(b_instance.to(A))

    c_instance = C(4, 3, 2, 1)
    print(b_instance.to(c_instance))