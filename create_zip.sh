#!/bin/bash

# Create zip file with specified contents
zip -r assignment_2.zip \
    output_plots/ \
    data/ \
    main/ \
    managed_components/ \
    sdkconfig \
    plot_imu.py \
    main/main.c \
    CMakeLists.txt \
    record_gestures.py \
    svm.py \
    plot_graphs.py \
    HW2.pdf