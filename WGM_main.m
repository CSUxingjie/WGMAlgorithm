%%%%%%%%%%%%%%% Topological optimization based on Novel Graph %%%%%%%%%%%%
function results = WGM_main
clc;clear all;
%%1 Topological definition
N = 5;             % number of vertices
M = N*(N-1)/2;     % number of maximal edges
T = 3;             % thick variation
nelx = 24;
nely = 12;
vol = 0.50;
%%2 Boundary Definition
% [C1 C2 ... CM P1 P2 ... PN];
lb = zeros(1,M+2*N);
ub = T*ones(1,M+2*N);
for i = 1:N
    ub(2*(i-1)+1+M) = nelx;
    ub(2*(i-1)+2+M) = nely;
end
% specific vertex definition
%  node lb_x ub_x lb_y ub_y  
sp = [1   24   24    6    6;
      2    0    0    0   12;];
for k = 1:size(sp,1)
    lb(2*(sp(k,1)-1)+1+M) = sp(k,2);
    ub(2*(sp(k,1)-1)+1+M) = sp(k,3);
    lb(2*(sp(k,1)-1)+2+M) = sp(k,4);
    ub(2*(sp(k,1)-1)+2+M) = sp(k,5);
end
%%3 Results record
results = zeros(5,size(lb,2)+2);
parfor i = 1:8
    [best_f,best_x,G] = Differential_Evolution(lb,ub,nelx,nely,N,vol);
    results(i,:) = [best_f,best_x,G];
    figure(i)
    best_map = Topodecode((best_x),nelx,nely,N);
    colormap(gray);imagesc(-best_map);axis equal;axis tight;axis off;pause(1e-6);
    saveas(figure(i),strcat('24X12(',num2str(i+8),')'),'jpg');
end
save('24X12(2)');

%%%%%%%%%%%%%%%%%%%%  Differential Evolution Algorithm %%%%%%%%%%%%%%%%%%%
function [best_f,best_x,G] = Differential_Evolution(lb,ub,nelx,nely,N,vol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 0.Definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dim = size(lb,2);            
F0 = 0.5;
Cr = 0.8;
GM = 10000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1.Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%
NP = 100;
X = zeros(NP,Dim);
for i = 1:NP
    for j = 1:Dim
        X(i,j) = randi([lb(j),ub(j)]);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2.Loop Optimization %%%%%%%%%%%%%%%%%%%%%%%%
G = 1;
[fit_X,con_X] = fitness_function(X,nelx,nely,N,vol);
Obj_X = constraint_judgement(fit_X,con_X);
[~,best_index]=min(Obj_X);
best_f = fit_X(best_index);
avg_f = sum(fit_X)/NP;
best_x = X(best_index,:);
while G <= GM
    %%%% 2.1 mutation
    Fg = (1+sin(pi/2*(G-0.5*GM)/(0.5*GM)))/2;
    Fc = exp(-1/abs(best_f-avg_f));
    V = mutation(X,best_x,Fg,Fc,3,lb,ub);
    %%%% 2.2 crossover
    U = crossover(X,V,Cr);
    %%%% 2.3 selection
    [fit_U,con_U] = fitness_function(U,nelx,nely,N,vol);
    [fit_X,con_X,X] = selection(fit_X,con_X,fit_U,con_U,X,U);
    %%%% 2.4 record
    Obj_X = constraint_judgement(fit_X,con_X);
    [~,best_index]=min(Obj_X);
    best_f = fit_X(best_index);
    best_x = X(best_index,:);
    best_map = Topodecode((best_x),nelx,nely,N);
    colormap(gray);imagesc(-best_map);axis equal;axis tight;axis off;pause(1e-6);
    disp(['It.:' sprintf('%4i',G) ...
        'Obj.:' sprintf('%10.4f',best_f) ...
        'Vol.:' sprintf('%6.3f',sum(sum(best_map))/(nelx*nely))]); 
    %%%% 2.5 convergency 
    avg_f = sum(fit_X)/NP;
    if abs(best_f-avg_f) <=0.000001
        break;
    end    
    G = G+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%% Mutation opration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function V = mutation(X,best_x,Fg,Fc,mutationStrategy,lb,ub)
[NP,Dim] = size(X);
V = X;
for i = 1:NP
    % Prepare 4 numbers which different from i and each other
    p = randperm(NP,4);
    r = p(1:3);
    equali = find(p==i);
    if ~isempty(equali) == 1
        r(equali) = p(4);
    end
    switch mutationStrategy
        case 1
            %mutationStrategy = 1:DE/rand/1;
            V(i,:) = X(r(1),:)+F*(X(r(2),:)-X(r(3),:));
        case 2
            %mutationStrategy = 2:DE/best/1;
            V(i,:) = best_x+F*(X(r(1),:)-X(r(2),:));
        case 3
            %mutationStrategy = 3:DE/rand-to-best/1;
            V(i,:) = X(i,:)+(1-Fc)*(best_x-X(i,:))+(1-Fg)*(X(r(1),:)-X(r(2),:));
        otherwise
            error('没有所指定的变异策略，请重新设定mutationStrategy的值');
    end
    % Boundary check
    for j = 1:Dim
        if V(i,j) > ub(j)||V(i,j) < lb(j)
            V(i,j) = randi([lb(j),ub(j)]);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Crossover opration %%%%%%%%%%%%%%%%%%%%%%%%%%
function U = crossover(X,V,Cr)
[NP,Dim] = size(X);
U = X;
for i=1:NP
    jRand = randi([1,Dim]);
    for j = 1:Dim
        k = rand;
        if k <= Cr||j == jRand
            U(i,j) = V(i,j);
        else
            U(i,j) = X(i,j);
        end
    end
end
U = round(U);

%%%%%%%%%%%%%%%%%%%%%%%%% Selection opration %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fit_X,con_X,X] = selection(fit_X,con_X,fit_U,con_U,X,U)
[NP,~] = size(X);
var_pool = [U;X];
fit_pool = [fit_U;fit_X];
con_pool = [con_U;con_X];
Obj_pool = constraint_judgement(fit_pool,con_pool);
[~,index] = sort(Obj_pool);
fit_X = fit_pool(index(1:NP));
con_X = con_pool(index(1:NP));
X = var_pool(index(1:NP,:),:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fitness function %%%%%%%%%%%%%%%%%%%%%%%%%%
function [fit_X,con_X] = fitness_function(X,nelx,nely,N,vol)
[NP,~] = size(X);
fit_X = zeros(NP,1);
con_X = zeros(NP,1);
for i = 1:NP
    var = X(i,:);
    [xmap] = Topodecode(var,nelx,nely,N);
    [c,v] = FEA(nelx,nely,xmap);
    if v <= vol
        con_X(i) = 0;
    else
        con_X(i) = v-vol;
    end
    fit_X(i) = c;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% Constraint judgement %%%%%%%%%%%%%%%%%%%%%%%%%
function [Obj_pool] = constraint_judgement(fit_pool,con_pool)
pool_size = size(fit_pool,1);
Obj_pool = zeros(pool_size,1);
index = find(con_pool == 0);
if isempty(index) ==1
    fitness_worst = 0;
else
    fitness_worst = max(fit_pool(index));
end

for i = 1:pool_size
    if con_pool(i)>0
        Obj_pool(i) = fitness_worst+con_pool(i);
    else
        Obj_pool(i) = fit_pool(i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%% NovelGraph representation %%%%%%%%%%%%%%%%%%%%%%%%
function [xmap] = Topodecode(var,nelx,nely,N)
%var = [C1 C2,...CM P1 P2,...PN]
xmap = 0.001*ones(nely,nelx);
M = round(N*(N-1)/2);
connect = Reshape_vector(var(1:M));
for i = 1:size(connect,1)
    rs = connect(i,1);
    re = connect(i,2);
    th = connect(i,3)-1;
    ps = [var(2*(rs-1)+1+M) var(2*(rs-1)+2+M)];
    pe = [var(2*(re-1)+1+M) var(2*(re-1)+2+M)];
    for u = 0:0.01:1
        ru = (1-u)*ps+u*pe;
        cur = ceil(ru);
        zb = max(1,cur(1)-th);  yb = min(cur(1)+th,nelx);
        xb = max(1,cur(2)-th);  sb = min(cur(2)+th,nely);
        xmap(xb:sb,zb:yb) = 1;
    end
end
%%symmitric constraint
for i = 1:nely
    for j = 1:nelx
        if xmap(i,j) == 1
            xmap(nely+1-i,j) = 1;
        end
    end
end

%%%%%%%%%%%%%%%%%%%% Reshape vector to connective matrix %%%%%%%%%%%%%%%%%
function connect = Reshape_vector(C)
n = (-1+sqrt(1+8*length(C)))/2+1;
A = 0.5*ones(n);
ind = find(triu(A,1));
A(ind) = C;
[m,n] = find(A>=1);
C(C==0) = [];
connect = [m,n,C'];

%%%%%%%%%%%%%%%%%%%%%% Finite element analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [c,vol] = FEA(nelx,nely,xmap)
%%1. Definition of K
E = 1.0;
nu = 0.3;
k = [1/2-nu/6 1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ...
    -1/4+nu/12 -1/8-nu/8 nu/6 1/8-3*nu/8];
KE = E/(1-nu^2)* ...
    [k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
     k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
     k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
     k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
     k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
     k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
     k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
     k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];
%%2. Initial set
K = zeros(2*(nelx+1)*(nely+1));
F = zeros(2*(nelx+1)*(nely+1),1);
U = zeros(2*(nelx+1)*(nely+1),1);
for ely = 1:nely
    for elx = 1:nelx
        n1 = (nely+1)*(elx-1)+ely;
        n2 = (nely+1)*elx+ely;
        edof = [2*n1-1;2*n1;2*n2-1;2*n2;2*n2+1;2*n2+2;2*n1+1;2*n1+2];
        K(edof,edof) = K(edof,edof)+xmap(ely,elx)^3*KE;
    end
end
%%3. Load and Constraint
F(2*nelx*(nely+1)+nely+2,1) = -1;
fixeddofs = 1:2*(nely+1);
alldofs = 1:2*(nely+1)*(nelx+1);
freedofs = setdiff(alldofs,fixeddofs);
%%4. Solve
U(freedofs,:) = K(freedofs,freedofs)\F(freedofs,:);
U(fixeddofs,:) = 0;
%%5. fitness calculation
c = 0;
for ely = 1:nely
    for elx = 1:nelx
        n1 = (nely+1)*(elx-1)+ely;
        n2 = (nely+1)*elx+ely;
        Ue = U([2*n1-1;2*n1;2*n2-1;2*n2;2*n2+1;2*n2+2;2*n1+1;2*n1+2],1);
        c = c+xmap(ely,elx)^3*Ue'*KE*Ue;
    end
end
vol = sum(sum(xmap))/(nelx*nely);
