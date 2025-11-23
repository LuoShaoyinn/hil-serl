export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=../../experiments/usb_pickup_insertion/debug \
    --demo_path=demo_data/data.pkl \
    --learner \

