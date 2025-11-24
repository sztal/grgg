from beartype import BeartypeConf, beartype

__all__ = ("validate",)


validate = beartype(conf=BeartypeConf(violation_type=ValueError, is_color=False))
