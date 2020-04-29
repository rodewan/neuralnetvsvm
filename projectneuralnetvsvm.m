%{
Rohit Dewan
%}
%{
for task 1 we first open the regression training file and copy the data to the x and t vectors, making a vector array for both variables for n observations,
wherein n is dynamically calculated based on the number of iterations of 15
entries, with 8 inputs and 7 outputs, that are detected.  


After reading the data, the built in MATLAB train command is used to train a neural network
with one hidden layer based on this training data set.
The training output is then compared with the training data
set and the least-squares error is calculated for each observation, summed up
over all observations, and normalized.

This process is repeated where the hidden layer has 1, 4, and 8 hidden units respectively
%}
%first we open the regression training file and copy the data to a cell type
fileID = fopen('regression.tra');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
regrxtrain=zeros(8,(length(newstr)/15));
regrttrain=zeros(7, (length(newstr)/15));
numobs=length(newstr)/15;
for x=1:numobs
    for y=1:8
        regrxtrain(y,x)=strread(newstr{((x-1)*15+y),1});
    end
    for z=1:7
        regrttrain(z,x)=strread(newstr{((x-1)*15+8+z),1});
    end    
end
%create neural networks with one, four, and eight hidden units, with the
%MLP function feedforwardnet, with the parameter specifying the number of
%hidden element units in the hidden layer
net=feedforwardnet(1);
net1=feedforwardnet(4);
net2=feedforwardnet(8);
[net,tr]=train(net,regrxtrain,regrttrain);
[net1,tr1]=train(net1,regrxtrain,regrttrain);
[net2,tr2]=train(net2,regrxtrain,regrttrain);
%we get the training output by applying our trained neural network to the
%input data
troutput=net(regrxtrain);
troutput1=net1(regrxtrain);
troutput2=net2(regrxtrain);
%we then calculate the training error
errorcounter=0;
errorcounter1=0;
errorcounter2=0;
for obcount=1:numobs
    for opcount=1:7
        errorcounter=(regrttrain(opcount,obcount)-troutput(opcount,obcount))^2;
        errorcounter1=(regrttrain(opcount,obcount)-troutput1(opcount,obcount))^2;
        errorcounter2=(regrttrain(opcount,obcount)-troutput2(opcount,obcount))^2;    
    end
end
%we normalize by dividing by the number of observations
errorcounter = errorcounter/numobs;
errorcounter1 = errorcounter1/numobs;
errorcounter2 = errorcounter2/numobs;
 disp(sprintf('Least-Squares Training Error for regression.tra over all observations with 1 hidden unit is %d for %d observations',errorcounter,numobs));
 disp(sprintf('Least-Squares Training Error for regression.tra over all observations with 4 hidden units is %d for %d observations',errorcounter1,numobs));
 disp(sprintf('Least-Squares Training Error for regression.tra over all observations with 8 hidden units is %d for %d observations',errorcounter2,numobs));
%we then get import the regression testing data and apply it to our trained
%neural networks
fileID = fopen('regression.tst');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
regrxtest=zeros(8,(length(newstr)/15));
regrttest=zeros(7, (length(newstr)/15));
numobs=length(newstr)/15;
for x=1:(length(newstr)/15)
    for y=1:8
        regrxtest(y,x)=strread(newstr{((x-1)*15+y),1});
    end
    for z=1:7
        regrttest(z,x)=strread(newstr{((x-1)*15+8+z),1});
    end    
end
%we get the training output by applying our trained neural network to the
%input data
tsoutput=net(regrxtest);
tsoutput1=net1(regrxtest);
tsoutput2=net2(regrxtest);
%we then calculate the training error
errorcounter=0;
errorcounter1=0;
errorcounter2=0;
for obcount=1:numobs
    for opcount=1:7
        errorcounter=(regrttest(opcount,obcount)-tsoutput(opcount,obcount))^2;
        errorcounter1=(regrttest(opcount,obcount)-tsoutput1(opcount,obcount))^2;
        errorcounter2=(regrttest(opcount,obcount)-tsoutput2(opcount,obcount))^2;    
    end
end
%we normalize by dividing by the number of observations
errorcounter = errorcounter/numobs;
errorcounter1 = errorcounter1/numobs;
errorcounter2 = errorcounter2/numobs;
disp(sprintf('Least-Squares Training Error for regression.tst over all observations with 1 hidden unit is %d for %d observations',errorcounter,numobs));
disp(sprintf('Least-Squares Training Error for regression.tst over all observations with 4 hidden units is %d for %d observations',errorcounter1,numobs));
disp(sprintf('Least-Squares Training Error for regression.tst over all observations with 8 hidden units is %d for %d observations',errorcounter2,numobs));

%we then move to task 2, to design a three layer neural network for
%classification, with 2 inputs and 2 classes, with 1, 2, and 4 hidden units
%in the hidden layer of our neural network.

%for task 2 we repeat essentially the same operation as task 1 for reading the data into the vector arrays,
%however we have to perform an extra step for classes to translate them from discrete to vector outputs, where '1'
%becomes [1 0] and '2' becomes [0 1] respectively.

fileID = fopen('classification.tra');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
xvec=zeros(2,(length(newstr)/3));
tvec=zeros(2,(length(newstr)/3));
classnum=zeros(1,(length(newstr)/3));
classnumcheck=zeros(1,(length(newstr)/3));
numobs=length(newstr)/3;
for x=1:(length(newstr)/3)
    for y=1:2
        xvec(y,x)=strread(newstr{((x-1)*3+y),1});
    end
    classnum(x)=strread(newstr{((x-1)*3+3),1});
    %based on the classnum value we do binary coding for the t vector
    if classnum(x)==1
        tvec(1,x)=1;
        tvec(2,x)=0;
    elseif classnum(x)==2
        tvec(1,x)=0;
        tvec(2,x)=1;
    else
    end
end
%create neural networks for pattern recognition (classification) with one, two, and four hidden units, with the
%MLP function patternnet, with the parameter specifying the number of
%hidden element units in the hidden layer
net=patternnet(1);
net1=patternnet(2);
net2=patternnet(4);
[net,tr]=train(net,xvec,tvec);
[net1,tr1]=train(net1,xvec,tvec);
[net2,tr2]=train(net2,xvec,tvec);
%we get the training output by applying our trained neural network to the
%input data
troutput=net(xvec);
troutput1=net1(xvec);
troutput2=net2(xvec);
%we translate the results into class number by assigning the index with the
%greater result as the class
tclasses=vec2ind(tvec);
classes=vec2ind(troutput);
classes1=vec2ind(troutput1);
classes2=vec2ind(troutput2);
%we then calculate the training accuracy
errorcounter=0;
errorcounter1=0;
errorcounter2=0;
for obcount=1:numobs
        if (tclasses(obcount)~=classes(obcount))
            errorcounter=errorcounter+1;
        end
        if (tclasses(obcount)~=classes1(obcount))
            errorcounter1=errorcounter1+1;
        end
        if (tclasses(obcount)~=classes2(obcount))
            errorcounter2=errorcounter2+1;
        end
end
 disp(sprintf('Training Accuracy for classification.tra over all observations with 1 hidden unit is %d percent for %d observations',(numobs-errorcounter)/numobs*100,numobs));
 disp(sprintf('Training Accuracy for classification.tra over all observations with 2 hidden units is %d percent for %d observations',(numobs-errorcounter1)/numobs*100,numobs));
 disp(sprintf('Training Accuracy for classification.tra over all observations with 4 hidden units is %d percent for %d observations',(numobs-errorcounter2)/numobs*100,numobs));
%having trained the neural networks we then import the testing data from
%classification.tst

fileID = fopen('classification.tst');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
xvec=zeros(2,(length(newstr)/3));
tvec=zeros(2,(length(newstr)/3));
classnum=zeros(1,(length(newstr)/3));
classnumcheck=zeros(1,(length(newstr)/3));
numobs=length(newstr)/3;
for x=1:(length(newstr)/3)
    for y=1:2
        xvec(y,x)=strread(newstr{((x-1)*3+y),1});
    end
    classnum(x)=strread(newstr{((x-1)*3+3),1});
    %based on the classnum value we do binary coding for the t vector
    if classnum(x)==1
        tvec(1,x)=1;
        tvec(2,x)=0;
    elseif classnum(x)==2
        tvec(1,x)=0;
        tvec(2,x)=1;
    else
    end
end

%we get the testing output by applying our trained neural network to the
%input data
troutput=net(xvec);
troutput1=net1(xvec);
troutput2=net2(xvec);
%we translate the results into class number by assigning the index with the
%greater result as the class
tclasses=vec2ind(tvec);
classes=vec2ind(troutput);
classes1=vec2ind(troutput1);
classes2=vec2ind(troutput2);
%we then calculate the training accuracy
errorcounter=0;
errorcounter1=0;
errorcounter2=0;
for obcount=1:numobs
        if (tclasses(obcount)~=classes(obcount))
            errorcounter=errorcounter+1;
        end
        if (tclasses(obcount)~=classes1(obcount))
            errorcounter1=errorcounter1+1;
        end
        if (tclasses(obcount)~=classes2(obcount))
            errorcounter2=errorcounter2+1;
        end
end
 disp(sprintf('Testing Accuracy for classification.tst over all observations with 1 hidden unit is %d percent for %d observations',(numobs-errorcounter)/numobs*100,numobs));
 disp(sprintf('Testing Accuracy for classification.tst over all observations with 2 hidden units is %d percent for %d observations',(numobs-errorcounter1)/numobs*100,numobs));
 disp(sprintf('Testing Accuracy for classification.tst over all observations with 4 hidden units is %d percent for %d observations',(numobs-errorcounter2)/numobs*100,numobs));

%{
for task 3 we repeat essentially the same operation as task 2, where we now
have 16 inputs and 10 classes, and we use 5, 10, and 13 hidden units in the
hidden layer of the MLP
%}
fileID = fopen('zipcode.tra');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
xvec=zeros(16,(length(newstr)/17));
tvec=zeros(10,(length(newstr)/17));
classnum=zeros(1,(length(newstr)/17));
classnumcheck=zeros(1,(length(newstr)/17));
numobs=length(newstr)/17;
for x=1:(length(newstr)/17)
    for y=1:17
        xvec(y,x)=strread(newstr{((x-1)*17+y),1});
    end
    classnum(x)=strread(newstr{((x-1)*17+17),1});
    %based on the classnum value we do binary coding for the t vector
    tvec(classnum(x),x)=1;
end

%create neural networks for pattern recognition (classification) with one, two, and four hidden units, with the
%MLP function patternnet, with the parameter specifying the number of
%hidden element units in the hidden layer
net=patternnet(5);
net1=patternnet(10);
net2=patternnet(13);
[net,tr]=train(net,xvec,tvec);
[net1,tr1]=train(net1,xvec,tvec);
[net2,tr2]=train(net2,xvec,tvec);
%we get the training output by applying our trained neural network to the
%input data
troutput=net(xvec);
troutput1=net1(xvec);
troutput2=net2(xvec);
%we translate the results into class number by assigning the index with the
%greater result as the class
tclasses=vec2ind(tvec);
classes=vec2ind(troutput);
classes1=vec2ind(troutput1);
classes2=vec2ind(troutput2);
%we then calculate the training accuracy
errorcounter=0;
errorcounter1=0;
errorcounter2=0;
for obcount=1:numobs
        if (tclasses(obcount)~=classes(obcount))
            errorcounter=errorcounter+1;
        end
        if (tclasses(obcount)~=classes1(obcount))
            errorcounter1=errorcounter1+1;
        end
        if (tclasses(obcount)~=classes2(obcount))
            errorcounter2=errorcounter2+1;
        end
end
 disp(sprintf('Training Accuracy for zipcode.tra over all observations with 5 hidden unit is %d percent for %d observations',(numobs-errorcounter)/numobs*100,numobs));
 disp(sprintf('Training Accuracy for zipcode.tra over all observations with 10 hidden units is %d percent for %d observations',(numobs-errorcounter1)/numobs*100,numobs));
 disp(sprintf('Training Accuracy for zipcode.tra over all observations with 13 hidden units is %d percent for %d observations',(numobs-errorcounter2)/numobs*100,numobs));
%having trained the neural networks we then import the testing data from
%zipcode.tst
fileID = fopen('zipcode.tst');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
xvec=zeros(16,(length(newstr)/17));
tvec=zeros(10,(length(newstr)/17));
classnum=zeros(1,(length(newstr)/17));
classnumcheck=zeros(1,(length(newstr)/17));
numobs=length(newstr)/17;
for x=1:(length(newstr)/17)
    for y=1:17
        xvec(y,x)=strread(newstr{((x-1)*17+y),1});
    end
    classnum(x)=strread(newstr{((x-1)*17+17),1});
    %based on the classnum value we do binary coding for the t vector
    tvec(classnum(x),x)=1;
end

%we get the testing output by applying our trained neural network to the
%input data
troutput=net(xvec);
troutput1=net1(xvec);
troutput2=net2(xvec);
%we translate the results into class number by assigning the index with the
%greater result as the class
tclasses=vec2ind(tvec);
classes=vec2ind(troutput);
classes1=vec2ind(troutput1);
classes2=vec2ind(troutput2);
%we then calculate the training accuracy
errorcounter=0;
errorcounter1=0;
errorcounter2=0;
for obcount=1:numobs
        if (tclasses(obcount)~=classes(obcount))
            errorcounter=errorcounter+1;
        end
        if (tclasses(obcount)~=classes1(obcount))
            errorcounter1=errorcounter1+1;
        end
        if (tclasses(obcount)~=classes2(obcount))
            errorcounter2=errorcounter2+1;
        end
end
 disp(sprintf('Testing Accuracy for zipcode.tst over all observations with 5 hidden unit is %d percent for %d observations',(numobs-errorcounter)/numobs*100,numobs));
 disp(sprintf('Testing Accuracy for zipcode.tst over all observations with 10 hidden units is %d percent for %d observations',(numobs-errorcounter1)/numobs*100,numobs));
 disp(sprintf('Testing Accuracy for zipcode.tst over all observations with 13 hidden units is %d percent for %d observations',(numobs-errorcounter2)/numobs*100,numobs));

 
 
 %{
for task 4 we repeat essentially the same operation as tasks 2 and 3, where we now
use the SVM classifier instead.  For task 2 we use the fitcsvm function
for binary classification utilizing kernel functions, and then for task 3
we use the fitcecoc function which is a combination of binary SVM
classifiers, where the default kernel function used is linear
%}

fileID = fopen('classification.tra');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
xvec=zeros(2,(length(newstr)/3));
tvec=zeros(2,(length(newstr)/3));
classnum=zeros(1,(length(newstr)/3));
classnumcheck=zeros(1,(length(newstr)/3));
numobs=length(newstr)/3;
for x=1:(length(newstr)/3)
    for y=1:2
        xvec(y,x)=strread(newstr{((x-1)*3+y),1});
    end
    classnum(x)=strread(newstr{((x-1)*3+3),1});
    %based on the classnum value we do binary coding for the t vector
    if classnum(x)==1
        tvec(1,x)=1;
        tvec(2,x)=0;
    elseif classnum(x)==2
        tvec(1,x)=0;
        tvec(2,x)=1;
    else
    end
end
%create neural networks for pattern recognition (classification) with one, two, and four hidden units, with the
%MLP function patternnet, with the parameter specifying the number of
%hidden element units in the hidden layer
tclasses=vec2ind(tvec);
gvec=transpose(xvec);
%the fitcsvm function uses where each ROW is an observation and in order to
%perform this we must transpose the xvec matrix, and where the class labels
%are not vectors but single indices, which we accomplish through the
%vec2ind function above
svmmodel=fitcsvm(gvec,tclasses);
%we get the training output in terms of class number by applying our trained svm model to the
%input data
troutput=predict(svmmodel,gvec);
%we then calculate the training accuracy
errorcounter=0;
for obcount=1:numobs
        if (tclasses(obcount)~=troutput(obcount))
            errorcounter=errorcounter+1;
        end
end
 disp(sprintf('Training Accuracy for classification.tra over all observations with the binary SVM classifier is %d percent for %d observations',(numobs-errorcounter)/numobs*100,numobs));

 
 %having trained the neural networks we then import the testing data from
%classification.tst

fileID = fopen('classification.tst');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
xvec=zeros(2,(length(newstr)/3));
tvec=zeros(2,(length(newstr)/3));
classnum=zeros(1,(length(newstr)/3));
classnumcheck=zeros(1,(length(newstr)/3));
numobs=length(newstr)/3;
for x=1:(length(newstr)/3)
    for y=1:2
        xvec(y,x)=strread(newstr{((x-1)*3+y),1});
    end
    classnum(x)=strread(newstr{((x-1)*3+3),1});
    %based on the classnum value we do binary coding for the t vector
    if classnum(x)==1
        tvec(1,x)=1;
        tvec(2,x)=0;
    elseif classnum(x)==2
        tvec(1,x)=0;
        tvec(2,x)=1;
    else
    end
end
gvec=transpose(xvec);
%we get the testing output by applying our trained neural network to the
%input data
troutput=predict(svmmodel,gvec);
%we translate the results into class number by assigning the index with the
%greater result as the class
tclasses=vec2ind(tvec);
%we then calculate the training accuracy
errorcounter=0;
for obcount=1:numobs
        if (tclasses(obcount)~=troutput(obcount))
            errorcounter=errorcounter+1;
        end
end
 disp(sprintf('Testing Accuracy for classification.tst over all observations with the binary SVM classifier is %d percent for %d observations',(numobs-errorcounter)/numobs*100,numobs));

 %we now perform task 3 using a multiclass SVM classifier, fitcecoc(X,Y),
 %which uses a combination of binary SVM classifiers, using a linear kernel
 %function by default
 
fileID = fopen('zipcode.tra');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
xvec=zeros(16,(length(newstr)/17));
tvec=zeros(10,(length(newstr)/17));
classnum=zeros(1,(length(newstr)/17));
classnumcheck=zeros(1,(length(newstr)/17));
numobs=length(newstr)/17;
for x=1:(length(newstr)/17)
    for y=1:17
        xvec(y,x)=strread(newstr{((x-1)*17+y),1});
    end
    classnum(x)=strread(newstr{((x-1)*17+17),1});
    %based on the classnum value we do binary coding for the t vector
    tvec(classnum(x),x)=1;
end
gvec=transpose(xvec);
tclasses=vec2ind(tvec);
svmmodel=fitcecoc(gvec,tclasses);
%we get the training output in terms of class number by applying our
%trained SVM classifier
%input data
troutput=predict(svmmodel,gvec);
%we then calculate the training accuracy
errorcounter=0;
for obcount=1:numobs
        if (tclasses(obcount)~=troutput(obcount))
            errorcounter=errorcounter+1;
        end
end
 disp(sprintf('Training Accuracy for zipcode.tra over all observations with the multiclass SVM classifier is %d percent for %d observations',(numobs-errorcounter)/numobs*100,numobs));
%having trained the SVM classifier we then import the testing data from
%zipcode.tst
fileID = fopen('zipcode.tst');
datastored = textscan(fileID, '%s');
fclose(fileID);
newstr=datastored{1,1};
xvec=zeros(16,(length(newstr)/17));
tvec=zeros(10,(length(newstr)/17));
classnum=zeros(1,(length(newstr)/17));
classnumcheck=zeros(1,(length(newstr)/17));
numobs=length(newstr)/17;
for x=1:(length(newstr)/17)
    for y=1:17
        xvec(y,x)=strread(newstr{((x-1)*17+y),1});
    end
    classnum(x)=strread(newstr{((x-1)*17+17),1});
    %based on the classnum value we do binary coding for the t vector
    tvec(classnum(x),x)=1;
end
gvec=transpose(xvec);
tclasses=vec2ind(tvec);
%we get the testing output in terms of class number by applying our trained neural network to the
%input data
troutput=predict(svmmodel,gvec);
%we then calculate the training accuracy
errorcounter=0;
for obcount=1:numobs
        if (tclasses(obcount)~=troutput(obcount))
            errorcounter=errorcounter+1;
        end
end
 disp(sprintf('Testing Accuracy for zipcode.tst over all observations with the multiclass SVM classifier is %d percent for %d observations',(numobs-errorcounter)/numobs*100,numobs));

