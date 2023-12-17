'''
@Time    : 2021/2/16 14:32
@Author  : 19045845
'''
import inspect
import warnings

from .misc import is_seq_of
from functools import partial
from tabulate import tabulate
from typing import Dict, Optional, Iterable, Tuple, Iterator


class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        else:
            assert is_seq_of(
                module_name,
                str), ('module_name should be either of None, an '
                       f'instance of str or list, but got {type(module_name)}')
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            'The old API of register_module(module, force=False) '
            'is deprecated and will be removed, please use the new API '
            'register_module(name=None, force=False, module=None) instead.')
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be a str, but got {type(name)}')

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    return obj_cls(**args)


class RegistryFunc(Iterable[Tuple[str, object]]):
    """ 注册函数，非 class """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert (name not in self._obj_map), \
            "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register_func(self, obj: object = None) -> Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, object]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


def build(cfg, registry, *args, **kwargs):
    """之前用的 build 方式，对输入的 cfg 进行注册"""
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'`cfg` must contain the key "type",but got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError(f'registry must be an tpcv.Registry object,but got {type(registry)}')
    cfg_ = cfg.copy()

    type_ = cfg_.pop("type")
    if isinstance(type_, str):
        module = registry.get(type_)
        if module is None:
            raise KeyError(f"{type_} is not in the {registry.name} registry")

    obj_cls = module(*args, **kwargs)
    return obj_cls


def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a config
            dict or a list of config dicts. If cfg is a list, a
            the built modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
            ]
        # todo 可以参照 pytorch 的源码写一个顺序放入的模块
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


