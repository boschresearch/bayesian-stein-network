import os
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import yaml

dirname = os.path.dirname(__file__)


class ListEnum(str, Enum):
    list = "list"
    sweep = "sweep"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


def load_yaml(path: str) -> dict:
    with open(path, "r") as stream:
        result = yaml.safe_load(stream)
        assert type(result) is dict
        return result


def save_yaml(opts_dict: dict, name: str, path: str = "", save_option: str = "w"):
    with open(os.path.join(path, name), save_option) as outfile:
        yaml.dump(opts_dict, outfile, default_flow_style=False)


def load_yaml_values(path: str) -> dict:
    simplified_dict = dict()
    exp_dict = load_yaml(path)
    for settings_key, settings_dict in exp_dict.items():
        simplified_dict[settings_key] = settings_dict["value"]
    return simplified_dict


def remove_keys(config_dict: Dict) -> Dict:
    simplified_dict = dict()
    for settings_key, settings_dict in config_dict.items():
        simplified_dict[settings_key] = {"value": deepcopy(settings_dict["value"])}
    return simplified_dict


def generate_run_configurations(
    default_config: Dict,
    experiment_dir: str,
    exp_name: str,
    config_folder: str = "yaml_config",
) -> Tuple[List[str], str]:
    raw_config = load_yaml(os.path.join(experiment_dir, exp_name))

    run_configs, run_configs_for_names = extract_dicts_from_options(raw_config)
    run_names = extract_run_name_list_from_config(run_configs_for_names)
    run_config_path = os.path.join(experiment_dir, config_folder)
    os.makedirs(run_config_path, exist_ok=True)
    save_run_configs_to_yaml(default_config, run_configs, run_names, run_config_path)
    return run_names, run_config_path


def save_run_configs_to_yaml(
    default_config: Dict,
    run_configs: Union[List[dict], List[Union[dict, Any]]],
    run_names: List[str],
    path: str = "",
):
    for run_config, name in zip(run_configs, run_names):
        save_run_config_to_yaml(default_config, name, path, run_config)


def save_run_config_to_yaml(
    default_config: Dict, name: str, path: str, run_config: Dict
):
    run_dict = default_config
    for key, value in run_config.items():
        run_dict[key] = value
    save_yaml(run_dict, name + ".yaml", path=path)


def extract_run_name_list_from_config(
    run_configs_for_run_names: List[Dict],
) -> List[str]:
    run_names = []
    for config in run_configs_for_run_names:
        name = config_dict_to_run_name(config)
        run_names.append(name)
    return run_names


def config_dict_to_run_name(config: Dict) -> str:
    """
    Converts a dictionary into a run name as used for naming the folders for experiments
    :param config: configuration dictionary which specifies the run name
    :return: name of folder
    """
    name = "run"
    for settings_key, value in config.items():
        name += f"_{settings_key}_{value}".replace(".", "_")
    return name


def extract_dicts_from_options(
    raw_config: Dict,
) -> Tuple[Union[List[dict], List[Union[dict, Any]]], List[dict]]:
    run_configs: List[dict] = [{}]
    run_configs_for_run_names: List[dict] = [{}]
    identifier_list: List[dict] = []
    run_configs, run_configs_for_run_names = extract_dict_sub_dict(
        raw_config, run_configs, run_configs_for_run_names, identifier_list
    )
    return run_configs, run_configs_for_run_names


def dict_at_list(d: Dict, l: List) -> Dict:
    for i in l:
        if i not in d.keys():
            d[i] = dict()
        d = d[i]
        if not isinstance(d, Dict):
            d = dict()
    return d


def extract_dict_sub_dict(
    raw_config: Dict,
    run_configs: List[Dict],
    run_configs_for_run_names: List[Dict],
    identifier_list: List,
):
    print(identifier_list)
    for settings_key, settings in raw_config.items():
        if not isinstance(settings, dict):
            for run_config in run_configs:
                dict_at_list(run_config, identifier_list)[settings_key] = settings
        elif ListEnum.has_value(value=list(settings.keys())[0]):
            run_configs, run_configs_for_run_names = extract_run_configs_from_list(
                run_configs,
                run_configs_for_run_names,
                settings,
                settings_key,
                identifier_list,
            )
        else:
            identifier_list.append(settings_key)
            run_configs, run_configs_for_run_names = extract_dict_sub_dict(
                raw_config=settings,
                run_configs=run_configs,
                run_configs_for_run_names=run_configs_for_run_names,
                identifier_list=identifier_list,
            )

    if len(identifier_list):
        identifier_list.pop()
    return run_configs, run_configs_for_run_names


def extract_run_configs_from_list(
    run_configs: List[Dict],
    run_configs_for_run_names: List[Dict],
    settings: Dict,
    settings_key: str,
    identifier_list: List,
):
    val_list = _extract_lists_from_yaml(settings_key, settings)
    print(val_list)
    new_run_configs = []
    new_run_name_configs = []
    for run_config, run_name_config in zip(run_configs, run_configs_for_run_names):
        dict_at_list(run_config, identifier_list)[settings_key] = val_list[0]
        new_run_configs.append(run_config)
        run_name_config[settings_key] = val_list[0]
        new_run_name_configs.append(run_name_config)
        for val in val_list[1:]:
            new_config = deepcopy(run_config)
            dict_at_list(new_config, identifier_list)[settings_key] = val
            new_run_configs.append(new_config)
            new_name_config = deepcopy(run_name_config)
            new_name_config[settings_key] = val
            new_run_name_configs.append(new_name_config)
    run_configs = new_run_configs
    run_configs_for_run_names = new_run_name_configs
    return run_configs, run_configs_for_run_names


def _extract_lists_from_yaml(settings_key: str, settings_dict: Dict) -> List:
    if ListEnum.list in settings_dict:
        val_list = settings_dict[ListEnum.list]
    elif ListEnum.sweep in settings_dict:
        if settings_dict[ListEnum.sweep].get("log"):
            val_list = np.geomspace(
                settings_dict[ListEnum.sweep]["start"],
                settings_dict[ListEnum.sweep]["stop"],
                settings_dict[ListEnum.sweep]["num"],
            )
            val_list = [val.item() for val in val_list]
        else:
            val_list = np.linspace(
                settings_dict[ListEnum.sweep]["start"],
                settings_dict[ListEnum.sweep]["stop"],
                settings_dict[ListEnum.sweep]["num"],
            )
            val_list = [val.item() for val in val_list]

    else:
        raise ValueError(
            f"No or wrong value for key {settings_key} supplied. Value: {settings_dict}"
        )
    assert isinstance(val_list, list)
    return val_list
