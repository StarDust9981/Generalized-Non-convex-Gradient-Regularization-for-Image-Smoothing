close all;
img = im2single(imread('flower.png'));

lambda=0.01;
alpha =-9999;
m =0.025;
sigma=0;
r=1;
iter=10;

result=MyFilter_GPU(img,lambda,alpha,m,sigma,r,iter);
figure;imshow([img,result]);
 