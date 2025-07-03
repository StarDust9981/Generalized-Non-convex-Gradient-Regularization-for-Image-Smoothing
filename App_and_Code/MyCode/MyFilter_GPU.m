function U =MyFilter_GPU(F, lambda, alpha, m, sigma, r, iter)

F = single(F); % 'single' precision is very important to reduce the computational cost/ single 4位，double 8�?
F=padarray(F,[1,1],0);%0Black, 1White


gamma = 0.5*alpha-1;
c = m^-2;

[N, M, D] = size(F);
sizeI2D = [N, M];

if sigma<=0 && r<=1
    P=F;
else
    if sigma==0
        sigma=0.01;
    end
    P1(:,:,1)=imgaussfilt(F(:,:,1),sigma,'Padding','symmetric');
    P1(:,:,2)=imgaussfilt(F(:,:,2),sigma,'Padding','symmetric');
    P1(:,:,3)=imgaussfilt(F(:,:,3),sigma,'Padding','symmetric');

    P(:,:,1)=medfilt2(P1(:,:,1),[r r],'symmetric');
    P(:,:,2)=medfilt2(P1(:,:,2),[r r],'symmetric');
    P(:,:,3)=medfilt2(P1(:,:,3),[r r],'symmetric');
end



otfFx = psf2otf_Dx_GPU(sizeI2D); % equal to otfFx = psf2otf(fx, sizeI2D) where fx = [1, -1];
otfFy = psf2otf_Dy_GPU(sizeI2D); % equal to otfFy = psf2otf(fy, sizeI2D) where fy = [1; -1];

Denormin = abs(otfFx).^2 + abs(otfFy ).^2;
Denormin = repmat(Denormin, [1, 1, D]);
Denormin = 1 + 0.5 * c * lambda * Denormin;

U = gpuArray(P); % smoothed image

Normin1 = fft2(U);

for k = 1: iter
    
    % Intermediate variables \mu update, in x-axis and y-axis direction
    u_h = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
    u_v = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
    
    if alpha>=2
        alpha=1.9999;
    end
    
    mu_h = c .* u_h - c .* u_h .* ((u_h .* u_h ) ./ ( m * m * abs( alpha - 2 )) + 1) .^ gamma;
    mu_v = c .* u_v - c .* u_v .* ((u_v .* u_v ) ./ ( m * m  * abs( alpha - 2 )) + 1) .^ gamma;
    
    % Update the smoothed image U
    Normin2_h = [mu_h(:,end,:) - mu_h(:, 1,:), - diff(mu_h,1,2)];
    Normin2_v = [mu_v(end,:,:) - mu_v(1, :,:); - diff(mu_v,1,1)];
    
    FU = (Normin1 + 0.5 * lambda * (fft2(Normin2_h + Normin2_v))) ./ Denormin;
    U = real(ifft2(FU));
    
    Normin1 = FU;  % This helps to further enlarge the smoothing strength
    
end
U=U(2:end-1,2:end-1,:);
U = gather(U);
end