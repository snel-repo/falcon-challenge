%choose training and validation sets:
num_obs = size(NeuralSig,1); %after duplicating for balance
K_train = KinematicSig; % Time x Dims
F_train = NeuralSig; % Time x Channels
K_val = KinematicSig_val;
F_val = NeuralSig_val;

invSigma = zeros(size(F_train,2),size(F_train,2),RTMA.defines.NUM_DOMAINS);
Kt = ControlSpaceToDomainCells(K_train);
Kv = ControlSpaceToDomainCells(K_val);
K = ControlSpaceToDomainCells([KinematicSig; KinematicSig_val]);
I = ControlSpaceToDomainCells(IgnoreIdx);
K_train = K_train(:,~IgnoreIdx);
K_val = K_val(:,~IgnoreIdx);

%calculate sigma matrix:
for i = 1:size(F_train,2)
    for d = 1:RTMA.defines.NUM_DOMAINS
        [~, ~, Residuals, ~, ~] = regress(single(F_train(:,i)), [ones(size(Kt{d},1), 1) Kt{d}(:,~I{d})]);
        Sigma = var(Residuals);
        invSigma(i,i,d) = 1/Sigma;
    end
end

%determine regularization coefficient(s) to use:
lambda = [0 logspace(-3,6,100)];
lambda1= [0 logspace(-3,6,100)];
idx1=1:length(lambda1);
idx = 1:length(lambda);
metric = inf(max(idx),max(idx1));
fprintf('\n')
counter('%d/%d',1,max(idx))
s = warning('off');

for b = idx1 %limit to a features
    if b>1 && (mod(b,10)==1 || b==max(idx1))
        counter('%d/%d',b,max(idx1))
    end
    W1 = ([ones(size(K_train,1), 1) K_train]'*[ones(size(K_train,1), 1) K_train] + lambda1(b)*eye(size(K_train,2)+1))\[ones(size(K_train,1), 1) K_train]'*F_train;
    baselines = W1(1,:);
    W1 = W1(2:end,:);
    repbaselines = repmat(baselines,[size(F_val,1) 1]); %for speed
    F_val_minus_baselines = (F_val-repbaselines);

    W1_ = zeros(size(W1,2),length(IgnoreIdx));
    W1_(:,~IgnoreIdx) = W1';
    W1 = ControlSpaceToDomainCells(W1_);
    W1 = cellfun(@(x,i) x(:,~i),W1,I,'UniformOutput',false);
    domains_to_use = find(~cellfun(@isempty,W1));
    w1invSigma = cell(1,max(domains_to_use));
    for d = domains_to_use
        w1invSigma{d} = W1{d}'*invSigma(:,:,d);
    end

    for a = idx
        W = [];
        for d = domains_to_use
            Wt = (pinv(w1invSigma{d}*W1{d} + lambda(a)*eye(size(W1{d}',1)))*w1invSigma{d})';
            W = [W Wt];
        end

        m = corrcoef(F_val_minus_baselines*W,K_val);
        metric(a,b) = m(1,2).^2; %norm(F_val*W-K_val);
    end

end

if sum(~isinf(metric))>1
    figure;semilogx(lambda,metric,'*-');
    xlabel('lambda'); ylabel('R2');
end

warning(s);

[v, reg_parm_to_use]=max(metric(:)); %lowest error / highest r2
[reg_parm_to_use, reg_parm_to_use1] = ind2sub(size(metric),reg_parm_to_use);

fprintf('Using lambda = %f, lambda1 = %f for %d observations... \n',lambda(reg_parm_to_use),lambda1(reg_parm_to_use1),size(NeuralSig,1))
lambda = lambda(reg_parm_to_use);
lambda1 = lambda1(reg_parm_to_use1);

%final train with full dataset:
if do_validation
    NeuralSig = [NeuralSig; NeuralSig_val];
    KinematicSig = [KinematicSig; KinematicSig_val];
end
KinematicSig = KinematicSig(:,~IgnoreIdx);

W = ([ones(size(KinematicSig,1), 1) KinematicSig]'*[ones(size(KinematicSig,1), 1) KinematicSig] + lambda1*eye(size(KinematicSig,2)+1))\[ones(size(KinematicSig,1), 1) KinematicSig]'*NeuralSig;
baselines = W(1,:);
W1 = W(2:end,:);
W = [];
W1_ = zeros(size(W1,2),length(IgnoreIdx));
W1_(:,~IgnoreIdx) = W1';
W1 = ControlSpaceToDomainCells(W1_);
W1 = cellfun(@(x,i) x(:,~i),W1,I,'UniformOutput',false);
domains_to_use = find(~cellfun(@isempty,W1));
for d = domains_to_use
    w1invSigma = W1{d}'*invSigma(:,:,d);
    Wt = (pinv(w1invSigma*W1{d} + lambda*eye(size(W1{d}',1)))*w1invSigma)';
    W = [W Wt];
end

weights = W;

%rescale weights so gain is reasonable:
scale = sqrt(mean((KinematicSig).^2,1)) ./ sqrt(mean(((NeuralSig-repmat(baselines,[size(NeuralSig,1) 1]))*weights).^2,1));
weights = weights .* (ones(size(weights,1),1)*scale);

obj.baselines = baselines;
obj.weights = weights;

%Report accuracy:
out = (NeuralSig-repmat(baselines,[size(NeuralSig,1) 1]))*weights;
r = corrcoef(out,KinematicSig);
normerr = norm(out-KinematicSig);


%scale back up if we have ignored any kinematic dims:
obj.weights = zeros(size(weights,1),length(IgnoreIdx));
obj.weights(:,~(IgnoreIdx==1)) = weights;
end
