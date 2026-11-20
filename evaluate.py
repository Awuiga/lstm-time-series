"""Оценить доверенный checkpoint и persistence baseline без переобучения."""
import argparse
import hashlib
import json
import platform
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        return self.regressor(output[:, -1, :])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='check_point.pt')
    parser.add_argument('--data',default='data/temperatures.csv')
    parser.add_argument('--output',default='reports/metrics.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    df=pd.read_csv(args.data)
    df['Date']=pd.to_datetime(df['Date'])
    df=df.sort_values('Date').reset_index(drop=True)
    if df[['Date','Temp']].isna().any().any() or df['Date'].duplicated().any():
        raise ValueError('Нужны непустые значения и уникальные даты.')
    train_size=int(len(df)*.7);val_size=int(len(df)*.15)
    frames={'train':df.iloc[:train_size], 'validation':df.iloc[train_size:train_size+val_size],
            'test':df.iloc[train_size+val_size:]}
    mean,std=cp['mean'],cp['std']
    if not np.isclose(mean,frames['train']['Temp'].mean()) or not np.isclose(std,frames['train']['Temp'].std()):
        raise ValueError('Mean/std checkpoint не соответствуют train.')
    model=TimeSeriesModel(cp['input_size'],cp['hidden_size']).to(device)
    model.load_state_dict(cp['model_state_dict'],strict=True)
    model.eval();window=cp['window_size']
    def score(actual,predicted):
        error=predicted-actual
        return {'rmse_celsius':float(np.sqrt(np.mean(error**2))),
                'mae_celsius':float(np.mean(np.abs(error)))}
    predictions={};results={}
    with torch.no_grad():
        for split in ['validation','test']:
            frame=frames[split]
            values=torch.tensor(((frame['Temp']-mean)/std).to_numpy(),dtype=torch.float32)
            if len(values)<=window:raise ValueError('Выборка короче окна.')
            inputs=torch.stack([values[i:i+window] for i in range(len(values)-window)]).unsqueeze(-1)
            output=torch.cat([model(batch.to(device)).cpu() for batch in inputs.split(64)]).squeeze(-1)
            actual=frame['Temp'].iloc[window:].to_numpy(dtype=np.float64)
            predicted=output.numpy().astype(np.float64)*std+mean
            baseline=frame['Temp'].iloc[window-1:-1].to_numpy(dtype=np.float64)
            dates=frame['Date'].iloc[window:].dt.strftime('%Y-%m-%d').tolist()
            predictions[split]={'dates':dates,'actual':actual.tolist(),'lstm':predicted.tolist(),'baseline':baseline.tolist()}
            results[split]={'rows':len(frame),'windows':len(actual),'target_start':dates[0],'target_end':dates[-1],
                            'lstm':score(actual,predicted),'baseline':score(actual,baseline)}
        latest=torch.tensor(((df['Temp'].tail(window)-mean)/std).to_numpy(),dtype=torch.float32).reshape(1,window,1)
        next_prediction=model(latest.to(device)).item()*std+mean
    report={'evaluation_date':datetime.now(timezone(timedelta(hours=3))).date().isoformat(),
            'rows':len(df),'period':[df['Date'].iloc[0].strftime('%Y-%m-%d'),df['Date'].iloc[-1].strftime('%Y-%m-%d')],
            'train_rows':train_size,'train_windows':train_size-window,
            'non_daily_gaps':int((df['Date'].diff().dropna()!=pd.Timedelta(days=1)).sum()),
            'configuration':{k:cp[k] for k in ['input_size','hidden_size','window_size','mean','std']},
            'parameters':sum(p.numel() for p in model.parameters()),**results,
            'next_observation_forecast_celsius':next_prediction,
            'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'csv_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'pandas':pd.__version__,'numpy':np.__version__,'device':device},
            'notes':['Saved user weights evaluated without training or tuning.',
                     'Chronological 70/15/15 split; windows formed separately inside each split.',
                     'Mean/std fit on train only; checkpoint values matched reconstructed train.',
                     'Rolling one-step evaluation uses true past temperatures, including past test observations.',
                     'Baseline is previous observed temperature on the exact same target dates.',
                     'Checkpoint does not store dataset hash, training epoch or split indices.',
                     'Latest forecast has no observed target in the supplied file.']}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    # График первых 100 целей test; исходный CSV и таблицы наблюдений не копируются в отчёт.
    snippet={k:v[:100] for k,v in predictions['test'].items()}
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    dates=pd.to_datetime(snippet['dates'])
    for key,label,color,style in [('actual','Фактическая температура','#334155','-'),
                                  ('lstm','LSTM','#2563eb','-'),
                                  ('baseline','Предыдущее наблюдение','#ea580c','--')]:
        ax.plot(dates,snippet[key],style,label=label,color=color,linewidth=1.4,alpha=.9)
    ax.set(title='Одношаговый прогноз — первые 100 целевых наблюдений test',
           xlabel='Дата прогнозируемого наблюдения',ylabel='Температура, °C')
    ax.grid(alpha=.2);ax.legend(loc='upper left')
    ax.tick_params(axis='x',labelrotation=20)
    fig.savefig(output.parent/'predictions.png',dpi=160)
    plt.close(fig)
    for split in ['validation','test']:print(split,results[split],flush=True)
    print('parameters',report['parameters'],'next forecast',next_prediction)


if __name__=='__main__':
    main()
