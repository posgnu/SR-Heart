# Super resolution imaging on CMR heart image

## Training
```sh
accelerate config
accelerate launch train.py
```

## Sampleing
```sh
python sample.py --weights 30 --sample-num 4
```