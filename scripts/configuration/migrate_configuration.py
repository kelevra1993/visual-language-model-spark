"""
When run, migrates an old run folder trained before the task-aware configuration and the backbone transfer to the
current configuration format through vision_training.configuration.configuration_migration.migrate_run_folder, and
prints (or applies) the new folder name given by the current run name, e.g. without the augmentation, batch size,
learning rate and dropout tokens of the older run names.
"""

from pathlib import Path

from vision_training.configuration.configuration_migration import migrate_run_folder

if __name__ == "__main__":
    # Run folder trained before the task-aware configuration and the backbone transfer, holding
    # 'training_configuration.yaml'
    run_folder = (Path(__file__).absolute().parents[2] / "experiments" /
                  "IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_"
                  "DSE-DRES-DBN-D7.2x64.256.128_BIN_AUG-BS40-LR2e-4")
    # Task the run was trained for
    task = "object_detection"
    # False only prints the new folder name, True renames the folder: never rename it while a training writes into it
    rename_folder = False

    migrate_run_folder(run_folder=run_folder, task=task, rename_folder=rename_folder)
