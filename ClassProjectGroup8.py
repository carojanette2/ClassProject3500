#adding this comment to test something
#Group 6: Matthew Quezada, Caroline Contreras, Avelina Olemedo, Christian Schmiedel

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from scipy.sparse import hstack
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def load_data(path):
    now = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
    try:
        start = datetime.now() 
        df = pd.read_csv(path)
        end = datetime.now()
        print(f"[{now}] Starting Script")
        print(f"[{now}] Loading training data set")
        print(f"[{now}] Total Columns Read: {df.shape[1]}")
        print(f"[{now}] Total Rows Read: {df.shape[0]}")
        print(f"\n Time to load is: {end - start}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        main()
    except pd.errors.EmptyDataError:
        print(f"[ERROR] No data: {path} is empty or corrupted")
        main()
    except pd.errors.ParserError:
        print(f"[ERROR] Parsing error: {path} may be malformed")
        main()
    except Exception as e:
        print(f"[ERROR] Unexpected error loading data: {e}")
        main()


def clean_data(df, write_intermediate=False):
    now = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
    try:
        start = datetime.now() 
        # 1) Set DR_NO as index
        df = df.set_index('DR_NO', drop=False)
        if write_intermediate:
            df.to_csv('after_set_index.csv')

        # 2) Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # 3) Convert types exactly like notebook
        df['Date Rptd'] = df['Date Rptd'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
        df['DATE OCC']  = df['DATE OCC'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        for col in ['AREA NAME','Crm Cd Desc','Mocodes','Vict Sex',
                    'Vict Descent','Premis Desc','Weapon Desc','Status','Status Desc']:
            if col in df.columns:
                df[col] = df[col].astype('string')

        # 4) Map Target
        mapping = {
            'IC': 'No Arrest','AA': 'Arrest','AO': 'No Arrest',
            'JO': 'No Arrest','JA': 'Arrest','CC': 'No Arrest'
        }
        df['Target'] = df['Status'].map(mapping)

        # 5) TIME OCC formatting
        df['TIME OCC'] = df['TIME OCC'].astype('string').str.zfill(4)
        df['TIME OCC'] = df['TIME OCC'].str[:-2] + ':' + df['TIME OCC'].str[-2:]
        df['TIME OCC'] = df['TIME OCC'].astype('string')

        # 6) Save intermediate cleaned file
        if write_intermediate:
            df.to_csv('df.csv', index=False)

        # 7) Remove duplicates
        df = df.drop_duplicates()

        # 8) Null fill for weapons
        df['Weapon Used Cd'] = df['Weapon Used Cd'].fillna(0)
        df['Weapon Desc']    = df['Weapon Desc'].fillna('No weapons identified')

        # 9) Filter Vict Age, Sex, Descent
        df = df[(df['Vict Age'] != 0) & (df['Vict Age'].notna())]
        df = df[~df['Vict Sex'].isin(['X','H']) & df['Vict Sex'].notna()]
        df = df[(df['Vict Descent'] != '-') & (df['Vict Descent'].notna())]

        # 10) Drop any remaining NaNs
        df = df.dropna()
        df = df.drop(df[df['Status'] == 'IC'].index)
        df = df.drop(df[df['Status'] == 'CC'].index)
        
        # 11) Dropping columns not used in this model
        columns_to_drop_not_used = ['Date Rptd', 'AREA NAME', 'Rpt Dist No', 'Part 1-2', 'Crm Cd Desc',
                                    'Premis Desc', 'Weapon Desc','Status', 'Status Desc']
        df.drop(columns=columns_to_drop_not_used, inplace=True)

        # 12) Feature engineering: cyclical time and date
        df['Hour'] = pd.to_datetime(df['TIME OCC'], format='%H:%M').dt.hour
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek'] = df['DATE OCC'].dt.dayofweek
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month'] = df['DATE OCC'].dt.month
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        end = datetime.now()

        print(f"[{now}] Performing Data Clean Up")
        print(f"[{now}] Total Rows after cleaning is: {df.shape[0]}")
        print(f"\n Time to proccess is: {end - start}")
        return df

    except KeyError as e:
        print(f"[ERROR] Missing column during cleaning: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error in cleaning: {e}")
        sys.exit(1)

1
def build_and_train(df):
    try:
        # Sample to reduce memory footprint
        df = df.sample(frac=0.1, random_state=42)

        # Numeric features including engineered ones
        num_feats = [
            'Vict Age','AREA','Crm Cd','Premis Cd','Weapon Used Cd',
            'Hour_sin','Hour_cos','DayOfWeek_sin','DayOfWeek_cos','Month_sin','Month_cos'
        ]
        cat_feats = ['Mocodes','Vict Sex','Vict Descent']

        # Remove outliers by IQR on raw numerics only
        raw_feats = ['Vict Age','AREA','Crm Cd','Premis Cd','Weapon Used Cd']
        Q1 = df[raw_feats].quantile(0.25)
        Q3 = df[raw_feats].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~(((df[raw_feats] < (Q1 - 1.5 * IQR)) |
                  (df[raw_feats] > (Q3 + 1.5 * IQR))).any(axis=1))
        df = df[mask]

        # Scale numeric features
        X_num = df[num_feats].values
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)

        # One-hot encode categorical
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_cat = encoder.fit_transform(df[cat_feats])

        # Combine features
        X = hstack([X_cat, X_num_scaled])

        # Target
        y = df['Target'].map({'No Arrest':0,'Arrest':1}).values

        # Split
        X_train_s, X_test_s, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Convert to dense for NN input
        X_train = X_train_s.toarray()
        X_test  = X_test_s.toarray()

        # Build model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())

        # Train (with early stopping)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=20, batch_size=32,
                  validation_split=0.1, callbacks=[es], verbose=1)

        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"[RESULT] Test loss: {loss:.4f}, accuracy: {acc:.4f}")

        preds = (model.predict(X_test) > 0.5).astype(int).flatten()
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("Classification Report:")
        print(classification_report(y_test, preds,
                                     target_names=['No Arrest','Arrest']))

        return model, encoder, scaler, y_test, preds

    except Exception as e:
        print(f"[ERROR] Training error: {e}")
        sys.exit(1)


def predict(model, encoder, scaler, df):
    try:
        num_feats = [
            'Vict Age','AREA','Crm Cd','Premis Cd','Weapon Used Cd',
            'Hour_sin','Hour_cos','DayOfWeek_sin','DayOfWeek_cos','Month_sin','Month_cos'
        ]
        cat_feats = ['Mocodes','Vict Sex','Vict Descent']
        X_cat = encoder.transform(df[cat_feats])
        X_num = scaler.transform(df[num_feats].values)
        X = hstack([X_cat, X_num])
        X_dense = X.toarray()
        preds = (model.predict(X_dense) > 0.5).astype(int).flatten()
        df['Prediction'] = np.where(preds==1, 'Arrest','No Arrest')
        return df[['DR_NO','Prediction']]
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Crime Status Prediction")
    parser.add_argument('--train', help="Path to training CSV")
    parser.add_argument('--test', help="Path to testing CSV")
    args = parser.parse_args()

    model = encoder = scaler = None
    train_df = test_df = None
    last_y_test = last_preds = None

    # The user must complete steps in order
    steps = ["1", "2", "3", "4", "5"]
    completedsteps = set()
    currentstep = 0

    def stepped(choice):
        nonlocal currentstep
        expected_choice = steps[currentstep]
        if choice == expected_choice:
            completedsteps.add(choice)
            currentstep += 1
            return True
        else:
            print(f"[ERROR] You must complete step ({expected_choice}) before choosing ({choice}).")
            return False

    while True:
        print("\nMenu:\n(1) Load training data\n(2) Clean training data\n(3) Train NN\n(4) Load testing data\n(5) Generate Predictions and Print Accuracy\n(6) Quit")
        choice = input("Select an option: ")

        if choice == '6':
            print('Exiting.')
            sys.exit(0)

        if choice in steps:
            if not stepped(choice):
                continue

        if choice == '1':
            path = args.train or input("Enter training file path: ")
            train_df = load_data(path)

        elif choice == '2':
            if train_df is None:
                print("[ERROR] Load training data first.")
                continue
            else:
                train_df = clean_data(train_df)

        elif choice == '3':
            if train_df is None:
                print("[ERROR] No data to train on.")
                continue
            else:
                model, encoder, scaler, last_y_test, last_preds = build_and_train(train_df)

        elif choice == '4':
            if model is None:
                print("[ERROR] Ensure data is trained before loading test data.")
                continue
            else:
                path = args.test or input("Enter testing file path: ")
                test_df = load_data(path)

        elif choice == '5':
            if model is None or test_df is None:
                print("[ERROR] Ensure model is trained and testing data is loaded.")
                continue
            else:
                test_df = clean_data(test_df)
                preds = predict(model, encoder, scaler, test_df)
                try:
                    preds.to_csv("C:/Users/caroj/OneDrive/Desktop/3500Project/predictionClassProject8.csv", index=False)
                    print("[INFO] Predictions saved to predictionClassProject8.csv")
                except Exception as e:
                    print(f"[ERROR] Could not save file: {e}")

                if last_y_test is None or last_preds is None:
                    print("[ERROR] No predictions available to compute accuracy.")
                else:
                    count = int((last_preds == last_y_test).sum())
                    pct = count / len(last_y_test) * 100
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{now}]  {count} of correct predicted observations.")
                    print(f"[{now}]  {pct:.2f}% of correct predicted observations.")

if __name__ == '__main__':
    main()




#source ~/cs3500_py311/bin/activate

#python "/Users/matthewquezada/Desktop/CS3500/ClassProjectGroup8.py"
   
#/Users/matthewquezada/Desktop/LA_Crime_Data_2023_to_Present_data.csv

#/Users/matthewquezada/Desktop/LA_Crime_Data_2023_to_Present_test1.csv
