#!/bin/bash

make preprocess BACKSTEPS=3

make preprocess BACKSTEPS=2

make preprocess BACKSTEPS=1 DROPOUT=1