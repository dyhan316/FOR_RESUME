import optuna

def objective(trial):
    x = trial.suggest_float('x', -3,3) #add parameter to optimize 
    
    return (x-2)**2

study = optuna.create_study()
study.optimize(objective, n_trials=20)

if __name__ == "__main__":    
    #make db_folder if needed
    db_folder = os.makedirs(f"./{args.save_path}/optuna_db", exist_ok = True)
    url = "sqlite:///" + os.path.join(os.getcwd(), "reproduce.db")

    # Add stream handler of stdout to show the messages => 이건 되어있던데 왜하는 건지 모르겠다 
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage = optuna.storages.RDBStorage(
        url = url,
        heartbeat_interval = 60,
        grace_period = 120    
    )
    
    study = optuna.create_study(study_name = "test_study_name", 
                                storage = storage,
                                load_if_exists = True,
                                direction = 'maximize') #pruner, sampler도 나중에 넣기) 
    study.optimize(main, n_trials = 10, timeout = 60000) #8 hrs : 28800
  