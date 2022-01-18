import os
import pickle

def data_saver(config,save_list,XorY):

    saveFlag = 'y'

    #count existing files
    count = 1
    df_dir = "./data/Train/DataFrames"
    
    

    filename = 'df_'+ XorY + str(count)

    for path in os.listdir(df_dir):
        if filename in os.path.join(df_dir, path):
            count += 1
            filename = 'df_'+ XorY + str(count)


        # if os.path.isfile(os.path.join(df_dir, path)) and XorY in os.path:
        #     count += 1

    

    

    for fname in os.listdir(df_dir):
        if XorY in fname:
            with open(os.path.join(df_dir, fname), 'rb') as f:
                print(os.path.join(df_dir, fname))
                pickle_list = pickle.load(f)

                if pickle_list[0] == config:
                    saveFlag = 'n'
                    while True:
                            saveFlag = input("Dataframe with current configuration already exists:  " + filename + "| Do you want to overwrite? (y/n) ")
                            if saveFlag not in ["y","n"]:
                                print("Sorry, please enter y/n:")
                                continue
                            else:
                                break

                    if saveFlag == 'y':
                        os.remove(os.path.join(df_dir,fname))
                        filename = fname

    if saveFlag == 'y':
        with open(os.path.join(df_dir, filename),'wb') as f:        
            pickle.dump(save_list,f)

def data_loader(load_config,XorY):
    df_directory = "./data/Train/DataFrames"
    config_match = False

    for filename in os.listdir(df_directory):
        with open(os.path.join(df_directory, filename), 'rb') as f:
            pickle_list = pickle.load(f)

            if pickle_list[0] == load_config and XorY in filename:
                print("Data loaded from ",os.path.join(df_directory, filename))
                all_stations = pickle_list[1]
                ind_stations = []
                
                for station in pickle_list[2:]:
                    ind_stations.append(station)
                    config_match = True

    if config_match == False:
        raise NameError('Configuration does not match any current dataframes. Check configuration or generate new data.')

    return all_stations,ind_stations
