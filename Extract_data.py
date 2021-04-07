import pandas as pd
import numpy as np
from random import randint
from scipy.signal import resample
import torch
from os import walk



def remap(value,classes):
    return np.where(classes==value)[0][0]

def remap_sorted(label,sorted_labels):
    return np.where(sorted_labels==label)[0][0]

def get_remapped_labels(data_df):
    sorted_labels=np.sort(np.unique(data_df['label']))
    return data_df['label'].apply(remap_sorted,args=[sorted_labels])

#interpolate missing data
#e.g. k=data_df['data'].apply(interpolate)
#this apply is in-place
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
def interpolate(d):
    nans, x= nan_helper(d)
    d[nans]= np.interp(x(nans), x(~nans), d[~nans])
    return d

#standardize data between 0 and 1
def get_standardized(d):
    maxs=np.max(d,axis=1)
    maxs=np.expand_dims(maxs,axis=1)
    maxs=np.repeat(maxs,d.shape[1],axis=1)
    mins=np.min(d,axis=1)
    mins=np.expand_dims(mins,axis=1)
    mins=np.repeat(mins,d.shape[1],axis=1)
    stand=(d-mins)/(maxs-mins)
    return stand

#normalize data (value-mean)/std
def get_normalized(d):
    means=np.mean(d,axis=1)
    means=np.expand_dims(means,axis=1)
    means=np.repeat(means,d.shape[1],axis=1)
    stds=np.std(d,axis=1)
    stds=np.expand_dims(stds,axis=1)
    stds=np.repeat(stds,d.shape[1],axis=1)
    stand=(d-means)/(stds)
    return stand

def get_global_standardized(d,max_vals,min_vals):
    max_vals=np.expand_dims(max_vals,axis=1)
    max_vals=np.repeat(max_vals,d.shape[1],axis=1)
    min_vals=np.expand_dims(min_vals,axis=1)
    min_vals=np.repeat(min_vals,d.shape[1],axis=1)
    stand=(d-min_vals)/(max_vals-min_vals)
    return stand

def get_chunks(d,sr,segment_len,step):
    chunks=np.array([d[:,i:i+segment_len] for i in range(0,d.shape[1]-segment_len,step)])
    return chunks
#sr-sample rate in Hz, segment_len and step-in seconds
def get_batch(df,bs,sr,segment_len,step):
    
    data=df.sample(n=bs)
    chunks=data['data'].apply(get_chunks,args=(sr,segment_len,step))

    labels=data['label'].values
    labels=labels.astype(int)
    return chunks,labels,data

#device data into train and test
def divide_data(data_df,ratio):
    train_data_len=len(data_df)*ratio
    randomized_df=data_df.sample(len(data_df))
    train_df=randomized_df.iloc[0:int(train_data_len)]
    test_df=randomized_df.iloc[int(train_data_len):]
    return train_df,test_df

'''
get cropped data suitable for a CNN with fixed input size
'''
def crop_data(data,segment_len):
    pad_len=segment_len-data.shape[1]
    if(pad_len>=0):
        data=np.pad(data,pad_width=((0,0),(0,pad_len)))
        return data
    start=np.random.randint(0,(data.shape[1]-segment_len))
    end=start+segment_len
    data=data[:,start:end]
    return data

#segment_len_s - segment length in seconds
#sr - sample rate 
def get_cropped_batch(df,sr,segment_len_s,bs):
    segment_len=segment_len_s*sr
    data=df.sample(n=bs)
    data['cropped']=data['data'].apply(crop_data,args=[segment_len])
    return data

#get a number of data samples larger than the length of the dataframe
#n_times is how many times of the dataframe is the length of the extracted data
def get_n_times_cropped_data(data_df,bs,n_times,sr=100,segment_len_s=10):
    df=get_cropped_batch(data_df,sr=sr,segment_len_s=segment_len_s,bs=bs)
    for i in range(n_times-1):
        tmp=get_cropped_batch(data_df,sr=sr,segment_len_s=segment_len_s,bs=bs)
        df=df.append(tmp)

    data=df['cropped']
    data=np.array([d for d in data])
    labels=df['label'].values

    labels=torch.from_numpy(labels)

    data=torch.from_numpy(data)
    data=data.float()   
    activity_vec=np.array([vec for vec in df['activity_vec'].values])
    activity_vec=torch.from_numpy(activity_vec)
    
    return data,labels,df['activity_name'],activity_vec

def get_test_acc_loss(model,criterion,data,labels):
    embedding,pred=model(data)
    predmax=torch.max(pred,1)[1]
    iscorrect=(predmax==labels)
    num_correct=torch.sum(iscorrect).item()
    num_total=iscorrect.shape[0]
    acc=num_correct/num_total
    loss=criterion(pred,labels).item()
    return acc,loss

def get_test_acc_by_class(model,data,labels):
    features,pred=model(data)
    predmax=torch.max(pred,1)[1]
    iscorrect=(predmax==labels)
    iscorrect=iscorrect.detach().cpu().numpy()
    labels=labels.detach().cpu().numpy()
    
    df=pd.DataFrame()
    df['iscorrect']=iscorrect
    df['label']=labels
    df=df.groupby(['label']).mean()
    return df

#extract data from http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
def get_UCI_data(ratio):
    #load data from memory
    df=pd.read_pickle('/scratch/lnw8px/activity_recognition/uci_data/12_Activities/raw_data.pkl')
    classes=np.array(list(range(1,13)))
    df['remapped']=df['activity'].apply(remap,args=[classes])
    df=df.drop(['activity'],axis=1)
    df=df.rename(columns={'remapped':'label'})
    train_len=len(df)*ratio
    randomized_df=df.sample(len(df))
    #standardize data
    #randomized_df['data']=randomized_df['data'].apply(get_standardized)
    
    #for globally standardizing
    #max_vals=np.max(np.array([val for val in randomized_df['data'].apply(np.max,axis=1)]),axis=0)
    #min_vals=np.min(np.array([val for val in randomized_df['data'].apply(np.min,axis=1)]),axis=0)
    #randomized_df['data']=randomized_df['data'].apply(get_global_standardized,args=(max_vals,min_vals))

    train_df=randomized_df.iloc[0:int(train_len)]
    test_df=randomized_df.iloc[int(train_len):]
    return train_df,test_df


#extract data from Ku-HAR dataset
'''
resample data
'''
original_sr=100
required_sr=50
def resample_data(signal,original_sr,required_sr):
    ratio=required_sr/original_sr
    return resample(signal,int(len(signal)*ratio),axis=0)

def get_ku_data():
    ku_df=pd.read_pickle('/scratch/lnw8px/activity_recognition/KU-HAR/extracted_data/Trimmed_raw_data.pkl')
    re=ku_df['data'].apply(resample_data,args=[original_sr,required_sr])
    ku_df['resampled']=re
    classes=np.unique(ku_df['activity'].values)
    print(classes)
    
    #transpose data because handling of earlier batch extraction funcitons
    tr=ku_df['resampled'].apply(np.transpose)
    ku_df['resampled']=tr

    ku_df=ku_df.drop(['data'],axis=1)
    ku_df=ku_df.rename(columns={'resampled':'data','activity':'remapped'})
    return ku_df


'''
extract data from UTWNETE found at https://www.utwente.nl/en/eemcs/ps/research/dataset/
paper Shoaib, Muhammad, Stephan Bosch, Ozlem Durmaz Incel, Hans Scholten, and Paul JM Havinga. "Complex human activity recognition using smartphone and wrist-worn motion sensors." Sensors 16, no. 4 (2016): 426.
'''
def get_UTWNETE_data():
    file='/scratch/lnw8px/activity_recognition/UTWENTE/UT_Data_Complex/smartphoneatwrist.csv'
    data_df=pd.read_csv(file,header=None)
    data_df=data_df.drop([0,1,2,3,10,11,12],axis=1)
    data_df.columns=['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','label']

    activity_labels=['11111','11112','11113','11114','11115','11116','11117','11118','11119','11120','11121','11122','11123']
    num_chunks=[10,10,10,10,10,10,10,7,7,7,7,6,7]

    data_list=[]
    for i in range(len(activity_labels)):
        d=data_df[data_df['label']==int(activity_labels[i])]
        chunks=np.array_split(d,num_chunks[i])
        for chunk in chunks:
            ar=chunk.drop(['label'],axis=1).values
            ar=np.transpose(ar)
            data_list.append([ar,i])
    data_list=np.array(data_list)
    df=pd.DataFrame(data_list)
    df.columns=['data','label']
    return df

def get_PAMAP2_Data():
    path1='/scratch/lnw8px/activity_recognition/PAMAP2/PAMAP2_Dataset/Protocol/'
    path2='/scratch/lnw8px/activity_recognition/PAMAP2/PAMAP2_Dataset/Optional/'
    
    col=['ts','activity','hr',
     'tmp_hand16',
     'acc16_x_hand','acc16_y_hand','acc16_z_hand',
     'acc6_x_hand','acc6_y_hand','acc6_z_hand',
     'gyr_x_hand','gyr_y_hand','gyr_z_hand',
     'mag_x_hand','mag_y_hand','mag_z_hand',
     'invalid','invalid','invalid','invalid',
     'tmp_chest',
     'acc16_x_chest','acc16_y_chest','acc16_z_chest',
     'acc6_x_chest','acc6_y_chest','acc6_z_chest',
     'gyr_x_chest','gyr_y_chest','gyr_z_chest',
     'mag_x_chest','mag_y_chest','mag_z_chest',
     'invalid','invalid','invalid','invalid',
     'tmp_ankle',
     'acc16_x_ankle','acc16_y_ankle','acc16_z_ankle',
     'acc6_x_ankle','acc6_y_ankle','acc6_z_ankle',
     'gyr_x_ankle','gyr_y_ankle','gyr_z_ankle',
     'mag_x_ankle','mag_y_ankle','mag_z_ankle',
     'invalid','invalid','invalid','invalid']
    
    select_col=['ts','activity',
                'acc16_x_hand','acc16_y_hand','acc16_z_hand',
                'gyr_x_hand','gyr_y_hand','gyr_z_hand',
               'acc16_x_chest','acc16_y_chest','acc16_z_chest',
                'gyr_x_chest','gyr_y_chest','gyr_z_chest',
                'acc16_x_ankle','acc16_y_ankle','acc16_z_ankle',
                'gyr_x_ankle','gyr_y_ankle','gyr_z_ankle',]
    
    '''
    select_col=['ts','activity',
                'acc16_x_hand','acc16_y_hand','acc16_z_hand',
                'gyr_x_hand','gyr_y_hand','gyr_z_hand']
    '''

    _, _, f = next(walk(path1))
    filenames1=[path1+file for file in f if (file.split('.')[-1]=='dat')]

    _, _, f = next(walk(path2))
    filenames2=[path2+file for file in f if (file.split('.')[-1]=='dat')]

    filenames=filenames1+filenames2
    
    data_list,activity_list,participant_list=[],[],[]
    for file in filenames:
        df=pd.read_csv(file,sep=' ',header=None)
        df.columns=col
        df=df[select_col]
        activities=np.unique(df['activity'])
        participant=file.split('/')[-1].split('.')[0][-2:]
        for activity in activities:
            selected=df[df['activity']==activity]
            if((selected['ts'].iloc[-1]-selected['ts'].iloc[0])<2):
                continue
            diffs=np.diff(selected['ts'].values)
            diffs=np.insert(diffs,0,diffs[0])
            selected['diff']=diffs
            selected=selected.reset_index(drop=True)
            break_indices=selected[selected['diff']>0.015].index
            break_indices=list(break_indices)+[selected.index.values[-1]]

            last_index=0
            time=0
            for i in break_indices:
                tmp_df=selected.iloc[last_index:i]
                last_index=i
                if(len(tmp_df)>100):
                    data_list.append(np.array(tmp_df.iloc[:,2:-1].T))
                    activity_list.append(tmp_df['activity'].iloc[0])
                    participant_list.append(participant)
                    time+=tmp_df['ts'].iloc[-1]-tmp_df['ts'].iloc[0]
    df=pd.DataFrame()
    df['data']=data_list
    df['label']=activity_list
    df['participant']=participant_list
    return df
