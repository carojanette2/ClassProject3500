import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.sparse import hstack
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"[ERROR] No data: {path} is empty or corrupted")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"[ERROR] Parsing error: {path} may be malformed")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error loading data: {e}")
        sys.exit(1)


def clean_data(df, write_intermediate=False):
    try:
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

        # 11) Dropping columns not used in this model
        columns_to_drop_not_used = ['Date Rptd', 'AREA NAME', 'Rpt Dist No', 'Part 1-2', 'Crm Cd Desc',
                                    'Premis Desc', 'Weapon Desc', 'Status Desc']
        df.drop(columns=columns_to_drop_not_used, inplace=True)

        print(f"[INFO] After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except KeyError as e:
        print(f"[ERROR] Missing column during cleaning: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error in cleaning: {e}")
        sys.exit(1)


def build_and_train(df):
    try:
        # Sample to reduce memory footprint
        df = df.sample(frac=0.1, random_state=42)

        num_feats = ['Vict Age','AREA','Crm Cd','Premis Cd','Weapon Used Cd']
        cat_feats = ['TIME OCC','Mocodes','Vict Sex','Vict Descent']

        # Remove outliers by IQR
        Q1 = df[num_feats].quantile(0.25)
        Q3 = df[num_feats].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~(((df[num_feats] < (Q1 - 1.5 * IQR)) |
                  (df[num_feats] > (Q3 + 1.5 * IQR))).any(axis=1))
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

        # Train (fewer epochs to speed up)
        model.fit(X_train, y_train, epochs=10, batch_size=32,
                  validation_split=0.1, verbose=1)

        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"[RESULT] Test loss: {loss:.4f}, accuracy: {acc:.4f}")

        preds = (model.predict(X_test) > 0.5).astype(int).flatten()
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("Classification Report:")
        print(classification_report(y_test, preds,
                                     target_names=['No Arrest','Arrest']))

        # Store test set and preds for accuracy reporting
        return model, encoder, scaler, y_test, preds

    except Exception as e:
        print(f"[ERROR] Training error: {e}")
        sys.exit(1)


def predict(model, encoder, scaler, df):
    try:
        num_feats = ['Vict Age','AREA','Crm Cd','Premis Cd','Weapon Used Cd']
        cat_feats = ['TIME OCC','Mocodes','Vict Sex','Vict Descent']
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
    parser = argparse.ArgumentParser(description="Crime Status Prediction Pipeline")
    parser.add_argument('--train', help="Path to training CSV")
    parser.add_argument('--test', help="Path to testing CSV")
    args = parser.parse_args()

    model = encoder = scaler = None
    train_df = test_df = None
    last_y_test = last_preds = None

    while True:
        print("\nMenu:\n1) Load training data\n2) Clean training data\n3) Train NN\n4) Load testing data\n5) Generate Predictions and Print Accuracy\n6) Quit")
        choice = input("Select an option: ")

        if choice=='1':
            path = args.train or input("Enter training file path: ")
            train_df = load_data(path)

        elif choice=='2':
            if train_df is None:
                print("[ERROR] Load training data first.")
            else:
                train_df = clean_data(train_df)

        elif choice=='3':
            if train_df is None:
                print("[ERROR] No data to train on.")
            else:
                model, encoder, scaler, last_y_test, last_preds = build_and_train(train_df)

        elif choice=='4':
            path = args.test or input("Enter testing file path: ")
            test_df = load_data(path)

        elif choice=='5':
            if model is None or test_df is None:
                print("[ERROR] Ensure model is trained and testing data is loaded.")
            else:
                test_df = clean_data(test_df)
                preds = predict(model, encoder, scaler, test_df)
                preds.to_csv('/Users/matthewquezada/Desktop/CS3500/predictionClassProject8.csv', index=False)
                print("[INFO] Predictions saved to predictionClassProject8.csv")
            if last_y_test is None or last_preds is None:
                print("[ERROR] No predictions available to compute accuracy.")
            else:
                count = int((last_preds == last_y_test).sum())
                pct = count / len(last_y_test) * 100
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{now}]  {count} of correct predicted observations.")
                print(f"[{now}]  {pct:.2f}% of correct predicted observations.")

        elif choice=='6':
            print("Exiting")
            break

        else:
            print("[ERROR] Invalid option.")

if __name__=='__main__':
    main()
