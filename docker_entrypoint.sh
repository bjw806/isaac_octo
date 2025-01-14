#! /bin/bash

wandb login $WANDB_API_KEY
echo 2 | exec tail -f /dev/null
