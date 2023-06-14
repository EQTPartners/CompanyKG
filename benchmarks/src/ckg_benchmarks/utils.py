"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import argparse
import json

import click
import random
from typing import Any, Callable

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set all relevant random seeds

    Args:
        seed (int): the seed to be used
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def ranged_type(value_type: type, min_value: Any, max_value: Any) -> Callable:
    """Return function handle of an argument type function for ArgumentParser checking a range:
    min_value <= arg <= max_value

    Args:
        value_type (type): value-type to convert arg to
        min_value (Any): minimum acceptable argument
        max_value (Any): maximum acceptable argument

    Returns:
        function: function handle of an argument type function for ArgumentParser
    """

    def range_checker(arg: str):
        try:
            f = value_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be a valid {value_type}")
        if f < min_value or f > max_value:
            raise argparse.ArgumentTypeError(
                f"must be within [{min_value}, {max_value}]"
            )
        return f

    # Return function handle to checking function
    return range_checker


class JsonDictParamType(click.ParamType):
    name = "json"

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            try:
                proc_val = json.loads(value)
            except json.decoder.JSONDecodeError as e:
                self.fail(f"{value!r} is not valid JSON: {e}", param, ctx)
            else:
                if not isinstance(proc_val, dict):
                    self.fail(f"JSON parameter '{value!r}' is not a dict", param, ctx)
                else:
                    return proc_val
        else:
            return value
