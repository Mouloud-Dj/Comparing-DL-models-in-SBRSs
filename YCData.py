#pip install rs_datasets
from rs_datasets import YooChoose
import pandas as pd
def get_yoochoose_100k():
    yc=YooChoose()
    data=yc.log
    data=data[:100000]
    data.dropna()
    data.columns = ['session_id', 'timestamp', 'item_id','category']
    data['timestamp']=pd.to_datetime(data['timestamp'])
    data = data.sort_values(['session_id', 'timestamp'],
                            ascending=[True, True])

    data['item_id_rank'] = data['item_id'].rank(method='dense').astype(int)

    # Resetting the 'item_id' column to start from 1 based on ranks
    data['item_id'] = data['item_id_rank']

    # Drop the temporary 'item_id_rank' column if needed
    data.drop('item_id_rank', axis=1, inplace=True)
    #id starts from 0
    data['item_id']=data.item_id-1

    return data

def create_sequences(df):
    df = df.sort_values(['session_id', 'timestamp'], ascending=True)
    session_ids = df['session_id'].unique()

    sequences = []
    labels = []

    for session_id in session_ids:
        session_items = df[df['session_id'] == session_id]['item_id'].values
        for i in range(2, len(session_items)):
            sequences.append(session_items[:i])
            labels.append(session_items[i])

    return sequences, labels

def get_sequences():
    sequences, labels = create_sequences(get_yoochoose_100k())
    return sequences, labels
