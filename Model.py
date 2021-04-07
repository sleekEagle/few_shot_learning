import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch import optim as optim
import torch.autograd as autograd
import Extract_data
import CenterLoss as cl
import torch.nn.functional as F

class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, num_classes,num_embeddings,num_features,in_channels,num_kernels):
        super().__init__()
        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=num_kernels, kernel_size=7,stride=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=3),
            #nn.Dropout(),
            nn.Conv1d(num_kernels, num_kernels, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            #nn.Dropout(),
            nn.Conv1d(num_kernels, num_kernels, 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        self.embedding= nn.Sequential(
            nn.Linear(num_features, num_embeddings),
            nn.ReLU()
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(num_embeddings, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        embedding=self.embedding(feature)
        out = self.classifier(embedding)
        return embedding,out
    
class ModelWork():
    def __init__(self,num_classes,num_embeddings,in_channels,sr,segment_len_s,num_kernels):
        self.sr=sr
        self.segment_len_s=segment_len_s
    
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
        is_cuda = torch.cuda.is_available()
        # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(self.device) 
        self.model=HARmodel(num_classes,num_embeddings,1,in_channels=in_channels,num_kernels=num_kernels)
            
        #figure out the feature dims
        data=torch.rand(1,in_channels,sr*segment_len_s)
        self.feature_dim=self.model.features(data).view(data.shape[0],-1).shape[1]
        self.model=HARmodel(num_classes,num_embeddings,self.feature_dim,in_channels,num_kernels=num_kernels)
        print('feature_dim='+str(self.feature_dim))
        self.model=self.model.to(self.device)
        self.criterion=nn.CrossEntropyLoss()
        self.lr=0.001
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
        self.criterion_center=cl.CenterLoss(num_classes,feat_dim=num_embeddings,use_gpu=True,device=self.device)
        self.optimizer_center=torch.optim.SGD(self.criterion_center.parameters(),lr=self.lr)
        self.wv_loss=nn.MSELoss()
        
        
        
    '''
    train with softmax labels + wv MSE +center loss
    '''
    def train(self,train_df,test_df,ALPHA=0.5,LAMBDA=0.0001,max_itr=20000,discount=0.9,bs=32):
        train_loss_list,test_loss_list,test_acc_list,train_word_loss_list,train_center_loss_list=[],[],[],[],[]
        max_acc=0
        for i in range(40000):
            if(i>max_itr):
                return max_acc,train_loss_list,train_word_loss_list,train_center_loss_list
            #prepare data
            #********************
            data,labels,activity_name,activity_vec=Extract_data.get_n_times_cropped_data(train_df,bs=bs,n_times=1,
                                                                           sr=self.sr,
                                                                           segment_len_s=self.segment_len_s)
            data=data.to(self.device)
            labels=labels.to(self.device)
            activity_vec=activity_vec.to(self.device)
            #***********************
            self.model.zero_grad()
            self.optimizer.zero_grad()
            self.optimizer_center.zero_grad()

            embedding,pred=self.model(data)
            label_loss=self.criterion(pred, labels)
            loss_center=self.criterion_center(embedding,labels)
            if(ALPHA>0):
                word_loss=self.wv_loss(activity_vec,embedding)
            else:
                word_loss=0

            total_loss=label_loss+LAMBDA*loss_center+ALPHA*word_loss
            total_loss.backward()
            self.optimizer.step()

            #to cancel the effect of LAMBDA when updating the centers
            if(LAMBDA>0):
                for param in self.criterion_center.parameters():
                    param.grad.data *= (1./LAMBDA)

                self.optimizer_center.step()

            train_loss_list.append(label_loss.item())
            train_center_loss_list.append(loss_center.item())


            if((i>0) and (i%100==0)):
                data,labels,activity_name,activity_vec=Extract_data.get_n_times_cropped_data(train_df,bs=bs,n_times=40,
                                                                           sr=self.sr,
                                                                           segment_len_s=self.segment_len_s)
                data=data.to(self.device)
                labels=labels.to(self.device)

                test_acc,test_loss=Extract_data.get_test_acc_loss(self.model,self.criterion,data,labels)
                
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                #print(str(test_acc)+'  '+str(test_loss))
                #reduce learning rate
                self.optimizer.param_groups[0]['lr']=self.optimizer.param_groups[0]['lr']*discount
                self.optimizer_center.param_groups[0]['lr']=self.optimizer_center.param_groups[0]['lr']*discount

                #save model
                if(test_acc>max_acc):
                    #torch.save(model.state_dict(),model_name)
                    max_acc=test_acc        
        