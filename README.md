# dp-beta

## installation

```bash
cd $CODE
git clone git@github.com:????
cd beta-bayes-ops 
virtualenv .env 
source .env/bin/activate
pip install --no-cache-dir requirements.txt
python -c "import cmdstanpy;cmdstanpy.install_cmdstan()"
```

or

```bash
cd ~/.cmdstan/cmdstan-2.32.0/
make build
```

#### test cmdstanpy

```bash
python -c "from cmdstanpy import CmdStanModel;betaBayes_stan = CmdStanModel(
    stan_file='stan_files/betaBayes_logisticRegression_ML.stan')"
```

## commands

### tune dpsgd

set `methods_args.method_names=['dpsgd']`

```bash
python src/main_sweep.py -m  +methods_args=bnn_sim ++methods_args.batch_size=2048 ++methods_args.method_names=\[\'dpsgd\'\] ++methods_args.lr=1e-2,1e-1,5e-1 ++methods_args.clip_norm=0.5,1,2 ++methods_args.epochs=10,20,30 sweeps.n_rep=1 wandb_args.project=dpsgd-tune-2
python src/main_sweep.py -m  +methods_args=bnn_sim ++methods_args.run_bnn=True ++methods_args.batch_size=2048 ++methods_args.method_names=\[\'dpsgd\'\] ++methods_args.lr=1e-2,1e-1,5e-1 ++methods_args.clip_norm=0.5,1,2 ++methods_args.epochs=10,20,30 sweeps.n_rep=1 wandb_args.project=dpsgd-tune-3
```

### debug

```bash
python src/main_attack.py -m methods_args.task_name=linReg methods_args.method_names=Minami attack.private=False attack.epsilon=100 attack.n_obs=2 methods_args.reg=0.5 methods_args.n_mcmc=3 attack.n_rounds=2 seed=0 methods_args.n_warmup=3 wandb_args.mode=offline
```

### attack sweep

```bash
python src/main_attack.py -m methods_args.task_name=linReg,logReg methods_args.method_names=Chaudhuri,Chaudhuri_sklearn,BetaBayes,Minami attack.private=False attack.epsilon=100 attack.n_obs=2 methods_args.reg=0.5 methods_args.n_mcmc=50 attack.n_rounds=1000 wandb_args.mode=online  
python src/main_attack.py -m methods_args.task_name=linReg,logReg methods_args.method_names=Chaudhuri,Chaudhuri_sklearn,BetaBayes,Minami attack.private=True attack.epsilon=100,10,1,0.1 attack.n_obs=2 methods_args.reg=0.5 methods_args.n_mcmc=1 attack.n_rounds=1000 wandb_args.mode=online 
python src/main_attack.py -m methods_args.task_name=mlpReg methods_args.method_names=Chaudhuri,BetaBayes,Minami attack.n_rounds=200 attack.private=False attack.epsilon=100 attack.n_obs=1,2,42 methods_args.n_mcmc=50 wandb_args.mode=online  
python src/main_attack.py -m methods_args.task_name=mlpReg methods_args.method_names=Chaudhuri,BetaBayes,Minami attack.n_rounds=200 attack.private=True attack.epsilon=100,10,1,0.1 attack.n_obs=1,2,42 methods_args.n_mcmc=1 wandb_args.mode=online
```
