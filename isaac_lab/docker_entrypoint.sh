#! /bin/bash

# exec python source/standalone/tutorials/00_sim/create_empty.py
# exec tail -f /dev/null
exec tensorboard --logdir . --host 0.0.0.0
