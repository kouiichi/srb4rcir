from srb.utils.registry import register_srb_tasks

from .task import Task, Task16Rcs, Task16RcsCfg, TaskCfg
from .task_visual import (
    VisualTask,
    VisualTask16Rcs,
    VisualTask16RcsCfg,
    VisualTaskCfg,
)

BASE_TASK_NAME = __name__.split(".")[-1]
register_srb_tasks(
    {
        BASE_TASK_NAME: {},
        f"{BASE_TASK_NAME}_16rcs": {
            "entry_point": Task16Rcs,
            "task_cfg": Task16RcsCfg,
        },
        f"{BASE_TASK_NAME}_visual": {
            "entry_point": VisualTask,
            "task_cfg": VisualTaskCfg,
        },
        f"{BASE_TASK_NAME}_16rcs_visual": {
            "entry_point": VisualTask16Rcs,
            "task_cfg": VisualTask16RcsCfg,
        },
    },
    default_entry_point=Task,
    default_task_cfg=TaskCfg,
)
