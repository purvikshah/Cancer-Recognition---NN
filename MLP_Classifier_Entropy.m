        % Program for Risk sensitive MLP..........................................
        
        clear all
        close all
        clc
        
        % Load the training data..................................................
        Ntrain=load('gcmtrain.dat');
        [no_of_samples,columns] = size(Ntrain);
        
        % Initialize the Algorithm Parameters.....................................
        inp = columns-14;          % No. of input neurons
        hid = 50;        % No. of hidden neurons
        %out = length(unique(Ntrain(:, end)));            % No. of Output Neurons
        out = 14;
        lam = 1.e-03;       % Learning rate
        epo = 3000; %0.73 accuracy achieved
        
       Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
        Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights
        misclass = zeros(epo,1);
            
        % Train the network.......................................................
            for ep = 1 : epo
              sumerr = 0;
              miscla = 0;
              for sa = 1 : no_of_samples 
                xx = Ntrain(sa,1:inp)';     % Current Sample
                tt = Ntrain(sa,inp+1:end)';
                for i=1:size(tt,1)
                    if(tt(i)==1)
                        ca = i;
                        break;
                    endif
                end
                Yh = 1./(1+exp(-Wi*xx));    % Hidden output
                Yo = Wo*Yh;               % Predicted output
                er = tt -Yo;
                %Wo, Wi need editing for cross entropy function.
                err = ((1 + tt)/2).*(1./(1+Yo)) - ((1 - tt)/2).*(1./(1-Yo));
                Wo = Wo + lam * (er * Yh'); % update rule for output weight'
                Wi = Wi + lam * ((Wo'*err).*Yh.*(1-Yh))*xx'; %update for input weight
                sumerr = sumerr + sum(er.^2);
                [~,cp] = max(Yo);           % Predicted class
                if ca~=cp 
                    miscla = miscla + 1;
                endif
              end
              misclass(ep) = miscla;
            end
            figure;
            plot(1:epo , misclass);

        % Test the network.........................................................
        NFeature=load('gcmtest.dat');
        [no_of_samples,~]=size(NFeature);
        res_tes = zeros(no_of_samples,1);
        conftes = zeros(out,out);
        for sa = 1: no_of_samples
                xx = NFeature(sa,1:inp)';   % Current Sample'
                Yh = 1./(1+exp(-Wi*xx));    % Hidden output
                ca = NFeature(sa,end);  % Actual class
                Yo = Wo*Yh;                 % Predicted output
                [~,cpp] = max(Yo);           % Predicted class
                %res_tes(sa,:) = cpp;
                conftes(ca,cpp) = conftes(ca,cpp) + 1;
                %[ca cpp]
        end
        conftes
        trace(conftes)/no_of_samples
        