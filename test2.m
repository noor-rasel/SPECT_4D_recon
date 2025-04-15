%% read SIMIND projection data
total_projections = zeros(64,64,64,8,8);
total_counts = 8e6; 
total_counts_rp1 = 8e6/8; 
blur = 1; % include the distance-dependent blur (PSF modelling)
OF_tag=0; % calculate MAP objective function, zero means "No"
load('roi.mat'); 
rng(20230110);
for resp = 0 : 7  
    for cardica = 0 : 7
        filename = sprintf('cardiac%d_tot_w1.a00',  cardica+1);
        fid = fopen(filename,'rb');
        simind = fread(fid,'float');
        fclose(fid);
        simind = reshape(simind,64,64,64);
        total_projections(:,:,:,cardica+1,resp+1) = simind;

    end
end

disp(size(total_projections));
total_projections=total_projections/sum(total_projections(:))*total_counts;
projections=total_projections(:,:,:,:,1); % for resp = 1
disp(size(projections));
for i = 1:size(projections, 4)
    for j = 1:size(projections, 3) 
        projections(:,:,j,i) = rot90(projections(:,:,j,i), 3);
    end
end

projections = random('poiss', projections);
projections(projections<0) = 0;
disp(['Counts for resp 1 : ', num2str(sum(projections(:)))]);

% % Display multiple slices for a fixed cardiac phase
% for n = 26:42
%     figure;
%     imagesc(projections(:,:,n,1));  
%     colorbar; colormap jet; 
%     title(['Transverse Slice ', num2str(n), ' - Cardiac Phase 1']);
%     pause;
% end

%% preparing system matrix using attenuation map

% predefined attenuation weight matrix
load weight64_mn        
% filename=sprintf("cardiac_avg_atn_%d.bin",1); % matched attenuation map
filename = "cardiac_avg_atn_resp_avg.bin"; % averaged attenuation map
fid=fopen(filename,'r');
x=fread(fid,64^3,'single');
fclose(fid);
x=reshape(x,[64 64 64]);
x = flip(x, 1);
x = flip(x, 3); 

wp_attnwgt=cell(4096,1); % Initializes attenuation weight cell array, 64 detectors angle and 64 pixel per angle (total 4096 detectors). 

% calculation of attenuation correction factors
for n=1:64
    if n<33
        wp_attnwgt((n-1)*64+1:n*64)=attnmat(x,n,wp_vray((n-1)*64+1:n*64),...
            wp_ipxl((n-1)*64+1:n*64),wp_wgt((n-1)*64+1:n*64)); % attnmat() calculates ac by ray-tracing algorithm  
        % Uses precomputed ray vectors (wp_vray), impact pixel indices (wp_ipxl), and weight factors (wp_wgt).
    else % Instead of computing new attenuation weights, it reuses symmetry from earlier projections (m = n - 32)
        m=n-32;
        wp_attnwgt((n-1)*64+1:n*64)=attnmat(x,n,wp_vray((m-1)*64+1:m*64),...
            wp_ipxl((m-1)*64+1:m*64),wp_wgt((m-1)*64+1:m*64));
    end
end

% Find Maximum Attenuation Factor per Angle
for j=1:64
    for i=1:64
        wp=wp_attnwgt{(j-1)*64+i}; % extract attenuation weights for detector pixel (i, j)
        wp_S(i)=max(sum(wp)); % sums the attenuation weights along the ray path and finds the peak ac factor for each pixel
    end
    wp_M(j)=max(wp_S); 
end 

% find maximum attenuation factor across all angles and normalize the
% attenuation weights for 4096 detector elements
for j=1:64
    for i=1:64
        wp_attnwgt{(j-1)*64+i}=wp_attnwgt{(j-1)*64+i}/max(wp_M);
    end
end 
save('weight64_attn1.mat','wp_attnwgt');

%% Noisy 3d recon
gbeta=0; % temproal regulization weight (0, no temproal smoothing)
sbeta=1e-4; 
sub_num=16; % or 8,  for too noisy images, reduce sub_num, increase it num 
it_num=10; % or 40
tic;
Im_maps=mbsrem4dv2(projections,repmat(roi,[1,1,64,8]),sub_num,it_num,OF_tag,...
     sbeta,gbeta,blur,1,0,0); 
toc;

disp(['3D noisy image data counts before norm: ', num2str(sum(Im_maps(:)))]);

Im_maps=Im_maps/sum(Im_maps(:)) * counts_rp1;

end
Im_maps(Im_maps<0)=0;
disp(['3D noisy image data counts after norm: ', num2str(sum(Im_maps(:)))]);
save('Im_maps_card_resp_avg_8e-6.mat', "Im_maps"); 


% Global coordinates (minx:maxx, miny:maxy, minz:maxz) defines the ROI (LV/heart region) inside the full 64×64×64 space. 
minx=16;maxx=41;
miny=15;maxy=43;
minz=26;maxz=42;

Im_maps_3D=Im_maps(minx:maxx,miny:maxy,minz:maxz,:);
disp(size(Im_maps_3D)); 

output_dir = '3D_recon_images_card_resp_avg_1e-4_8e6'; 
if ~exist(output_dir, 'dir')
    mkdir(output_dir); 
end

% Save the recon slice (slice index 36 corresponds to 11 in cropped volume)
for n=1:8
    fig = figure('Visible', 'off');
    imagesc(Im_maps_3D(:,:,11,n));  
    colorbar; colormap jet; 
    title(['Noisy 3D transverse slice 36 for Cardiac Phase-', num2str(n)]); 
    ax = gca;  % Get current axes
    ax.Toolbar = [];  % Remove toolbar
    filename = sprintf('%s/recon_phase_%d.png', output_dir, n);
    saveas(fig, filename); 
    close(fig); 
end

save('Im_maps_card_resp_avg_3D_8e6.mat', "Im_maps_3D")


% Optimizing sbeta values in 4d using both RMSE and SSIM
yin=zeros(size(Im_maps));
G=8;
for g=1:G
    yin(:,:,:,g)=convn(Im_maps(:,:,:,g),ones(3,3,3)/27,'same');
end
M1=zeros(64,64,64,3,G);
for i=1:G-1
    [vx,vy,vz]=motionele3d(yin(:,:,:,i:i+1),50,01);
    M1(:,:,:,:,i)=cat(4,vx,vy,vz);
end
[vx,vy,vz]=motionele3d(yin(:,:,:,[G 1]),50,01);
M1(:,:,:,:,G)=cat(4,vx,vy,vz);
for g=1:G
    ind5(g,:)=mod((g-2:g+2),G);
end
ind5(ind5==0)=G;
for gate=1:G
    MM=mf2matrix3d_5g(M1(:,:,:,:,ind5(gate,:)),1);
    filename=['n4dMM' num2str(gate)];
    save(filename,'MM');
end
% Optimizing sbeta 
gbeta = 3e-4; % Temporal regularization weight (0 = no temporal smoothing)
sub_num = 16; % Reduce for noisy images, increase for stability
it_num = 10;  % Increase for better reconstruction

sbeta_values = [0,1e-5,3e-5,5e-5,7e-5,1e-4,3e-4,5e-4,7e-4]; 
rmse_values = zeros(size(sbeta_values)); 
ssim_values = zeros(size(sbeta_values)); 
best_rmse = Inf;  
best_ssim = -Inf;  
best_sbeta_rmse = NaN;
best_sbeta_ssim = NaN;

% Load ground truth
load('Im_maps_truth_8e6.mat');

% Define output directory
output_dir = 'Optimiz_sb_in_gb_sb3d_1e-4_8e6';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);  
end

for i = 1:length(sbeta_values)
    sbeta = sbeta_values(i);
    fprintf('Testing sbeta = %.1e (%d/%d)\n', sbeta, i, length(sbeta_values));

    tic;
    Im_maps = mbsrem4dv2(projections, repmat(roi, [1,1,64,8]), ...
        sub_num, it_num, OF_tag, sbeta, gbeta, blur, 1, 0, 0);
    toc;

    Im_maps(Im_maps < 0) = 0;
    Im_maps = Im_maps / sum(Im_maps(:)) * total_counts;

    % Define ROI (heart region)
    minx=16; maxx=41;
    miny=15; maxy=43;
    minz=26; maxz=42;

    % Extract heart region
    Im_maps_actual = Im_maps(minx:maxx, miny:maxy, minz:maxz, :); 

    % Save the recon slice (slice index 36 corresponds to 11 in cropped volume)
    fig = figure('Visible', 'off');  
    imagesc(Im_maps_actual(:,:,11,1));  
    colorbar; 
    colormap jet;
    title(['Noisy 4D recon - transverse slice 36 for phase 1 (sbeta = ', num2str(sbeta), ')']);
    ax = gca;  % get current axis  
    ax.Toolbar = [];  % remove toolbar 
    filename = sprintf('%s/recon_sbeta_%.1e.png', output_dir, sbeta);
    saveas(fig, filename);
    close(fig);  

    % Compute RMSE
    rmse_values(i) = sqrt(mean((Im_maps_actual(:,:,:,1) - Im_maps_truth(:,:,:,1)).^2, 'all'));

    % Compute SSIM 
    ssim_values(i) = ssim(Im_maps_actual(:,:,:,1), Im_maps_truth(:,:,:,1));

    % Find best RMSE
    if rmse_values(i) < best_rmse
        best_rmse = rmse_values(i);
        best_sbeta_rmse = sbeta;
    end

    % Find best SSIM
    if ssim_values(i) > best_ssim
        best_ssim = ssim_values(i);
        best_sbeta_ssim = sbeta;
    end

    close(gcf);
end

% Display best results
fprintf('\nBest sbeta (RMSE): %.1e with RMSE = %.2f\n', best_sbeta_rmse, best_rmse);
fprintf('Best sbeta (SSIM): %.1e with SSIM = %.2f\n', best_sbeta_ssim, best_ssim);

% Plot RMSE and SSIM vs sbeta
figure;
yyaxis left  
semilogx(sbeta_values, rmse_values, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'RMSE'); 
ylabel('RMSE');  
hold on;
plot(best_sbeta_rmse, best_rmse, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10, 'DisplayName', 'Best RMSE');

yyaxis right 
semilogx(sbeta_values, ssim_values, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SSIM');
ylabel('SSIM');
plot(best_sbeta_ssim, best_ssim, 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 10, 'DisplayName', 'Best SSIM');

% Annotate best RMSE and SSIM values
yyaxis left
text(best_sbeta_rmse, best_rmse, sprintf('%.1e, %.2f', best_sbeta_rmse, best_rmse), ...
'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'r');

yyaxis right
text(best_sbeta_ssim, best_ssim, sprintf('%.1e, %.2f', best_sbeta_ssim, best_rmse), ...
'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'b');

% Labels and Titles
xlabel('sbeta (log scale)');
title('RMSE and SSIM vs. sbeta');

% Legend
legend('Location', 'best');

% Grid for better visualization
grid on;
hold off;

% Save the figure
saveas(gcf, sprintf('%s/RMSE_SSIM_vs_sbeta.png', output_dir));
close(gcf);









% 4d recon with optimized values
yin=zeros(size(Im_maps));
G=8;
for g=1:G
    yin(:,:,:,g)=convn(Im_maps(:,:,:,g),ones(3,3,3)/27,'same');
end
M1=zeros(64,64,64,3,G);
for i=1:G-1
    [vx,vy,vz]=motionele3d(yin(:,:,:,i:i+1),50,01);
    M1(:,:,:,:,i)=cat(4,vx,vy,vz);
end
[vx,vy,vz]=motionele3d(yin(:,:,:,[G 1]),50,01);
M1(:,:,:,:,G)=cat(4,vx,vy,vz);
for g=1:G
    ind5(g,:)=mod((g-2:g+2),G);
end
ind5(ind5==0)=G;
for gate=1:G
    MM=mf2matrix3d_5g(M1(:,:,:,:,ind5(gate,:)),1);
    filename=['n4dMM' num2str(gate)];
    save(filename,'MM');
end
% 4D recon
gbeta=3e-4; 
sbeta=0; 
sub_num=16;% or 8, for too noisy images, reduce sub_num and increase it num 
it_num=10;% or 40 
tic;
Im_mapst=mbsrem4dv2(projections,repmat(roi,[1,1,64,8]),sub_num,it_num,OF_tag,...
    sbeta,gbeta,blur,1,0,0);
toc;

disp(['4D noisy image data counts before norm: ', num2str(sum(Im_mapst(:)))]);

Im_mapst=Im_mapst/sum(Im_mapst(:)) * total_counts;
Im_mapst(Im_mapst<0)=0;
disp(['4D noisy image data counts after norm: ', num2str(sum(Im_mapst(:)))]);
save('Im_mapst_8e6.mat', "Im_mapst");

% Global coordinates (minx:maxx, miny:maxy, minz:maxz) defines the ROI (LV/heart region) inside the full 64×64×64 space
minx=16;maxx=41;
miny=15;maxy=43;
minz=26;maxz=42;

Im_maps_4D=Im_mapst(minx:maxx,miny:maxy,minz:maxz,:);
disp(size(Im_maps_4D));
save('Im_maps_4D_8e6.mat', "Im_maps_4D");

output_dir = '4D_recon_images_gb_0.0003_sb_1e-4_8e6'; 
if ~exist(output_dir, 'dir')
    mkdir(output_dir); 
end

% % Save the recon slice (slice index 36 corresponds to 11 in cropped volume)
for n=1:8
    fig = figure('Visible', 'off');  
    imagesc(Im_maps_4D(:,:,11,n));  
    colorbar; colormap jet; 
    title(['Noisy 4D recon transverse slice 36 for Cardiac Phase-', num2str(n)]);
    ax = gca;  % Get current axes
    ax.Toolbar = [];  % Remove toolbar
    filename = sprintf('%s/recon_phase_%d.png', output_dir, n);
    saveas(fig, filename);
    close(fig);  
end




% Optimization of 4d recon gbeta values using SSIM and RMSE
yin=zeros(size(Im_maps));
G=8;
for g=1:G
    yin(:,:,:,g)=convn(Im_maps(:,:,:,g),ones(3,3,3)/27,'same');
end
M1=zeros(64,64,64,3,G);
for i=1:G-1
    [vx,vy,vz]=motionele3d(yin(:,:,:,i:i+1),50,01);
    M1(:,:,:,:,i)=cat(4,vx,vy,vz);
end
[vx,vy,vz]=motionele3d(yin(:,:,:,[G 1]),50,01);
M1(:,:,:,:,G)=cat(4,vx,vy,vz);
for g=1:G
    ind5(g,:)=mod((g-2:g+2),G);
end
ind5(ind5==0)=G;
for gate=1:G
    MM=mf2matrix3d_5g(M1(:,:,:,:,ind5(gate,:)),1);
    filename=['n4dMM' num2str(gate)];
    save(filename,'MM');
end

% Optimizing gbeta values
sbeta = 0; 
sub_num = 16; % Reduce for noisy images
it_num = 10;  % Increase for better reconstruction
gbeta_values = [0,1e-5,3e-5,5e-5,7e-5,1e-4,3e-4,5e-4,7e-4,1e-3,5e-3]; 
rmse_values = zeros(size(gbeta_values)); 
ssim_values = zeros(size(gbeta_values)); 
best_rmse = Inf;  
best_ssim = -Inf;  
best_gbeta_rmse = NaN;
best_gbeta_ssim = NaN;

% Define output directory
output_dir = 'Optimiz_gbeta_sb_3d_1e-4_8e6';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);  
end 

% Load ground truth
load("Im_maps_truth_8e6.mat");

for i = 1:length(gbeta_values)
    gbeta = gbeta_values(i); 
    fprintf('Testing gbeta = %e (%d/%d)\n', gbeta, i, length(gbeta_values));

    tic;
    Im_mapst = mbsrem4dv2(projections, repmat(roi, [1,1,64,8]), sub_num, it_num, OF_tag, ...
                          sbeta, gbeta, blur, 1, 0, 0);
    toc;

    Im_mapst(Im_mapst < 0) = 0;
    Im_mapst = Im_mapst / sum(Im_mapst(:)) * total_counts;

    % Define ROI (heart region)
    minx=16; maxx=41;
    miny=15; maxy=43;
    minz=26; maxz=42;

    % Extract heart region
    Im_maps_4D = Im_mapst(minx:maxx, miny:maxy, minz:maxz, :);

    % save recon slice (slice index 36 corresponds to 11 in cropped volume)
    fig = figure('Visible','off');
    imagesc(Im_maps_4D(:,:,11,1));  
    colorbar; colormap jet;
    title(['Noisy 4D recon - transverse slice 36 for phase 1 (gbeta = ', num2str(gbeta), ')']);
    ax = gca;  % get current axis
    ax.Toolbar = [];  % remove toolbar
    filename = sprintf('%s/recon_gbeta_%e.png', output_dir, gbeta);
    saveas(fig, filename);
    close(fig);

    % Compute RMSE
    rmse_values(i) = sqrt(mean((Im_maps_4D(:,:,:,1) - Im_maps_truth(:,:,:,1)).^2, 'all'));

    % Compute SSIM 
    ssim_values(i) = ssim(Im_maps_4D(:,:,:,1), Im_maps_truth(:,:,:,1));

    % Find best RMSE
    if rmse_values(i) < best_rmse
        best_rmse = rmse_values(i);
        best_gbeta_rmse = gbeta;
    end

    % Find best SSIM
    if ssim_values(i) > best_ssim
        best_ssim = ssim_values(i);
        best_gbeta_ssim = gbeta;
    end

    close(gcf);
end

% Display best results
fprintf('\nBest RMSE: %.2f at gbeta = %.1e\n', best_rmse, best_gbeta_rmse);
fprintf('Best SSIM: %.2f at gbeta = %.1e\n', best_ssim, best_gbeta_ssim);

% Plot RMSE and SSIM vs gbeta
figure;
yyaxis left  
semilogx(gbeta_values, rmse_values, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'RMSE'); 
ylabel('RMSE');  
hold on;
plot(best_gbeta_rmse, best_rmse, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10, 'DisplayName', 'Best RMSE');

yyaxis right 
semilogx(gbeta_values, ssim_values, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SSIM');
ylabel('SSIM');
plot(best_gbeta_ssim, best_ssim, 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 10, 'DisplayName', 'Best SSIM');

% Annotate best RMSE and SSIM values
yyaxis left
text(best_gbeta_rmse, best_rmse, sprintf('%.1e, %.2f', best_gbeta_rmse, best_rmse), ...
'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'r');

yyaxis right
text(best_gbeta_ssim, best_ssim, sprintf('%.1e, %.2f', best_gbeta_ssim, best_rmse), ...
'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'b');

% Labels and Titles
xlabel('gbeta (log scale)');
title('RMSE and SSIM vs. gbeta');

% Legend
legend('Location', 'best');

% Grid for better visualization
grid on;
hold off;

% Save the figure
saveas(gcf, sprintf('%s/RMSE_SSIM_vs_gbeta.png', output_dir));
close(gcf);






% Visualization of motion flow on 3d recon images

% spatial smooting using 3d convolution
yin=zeros(size(Im_maps_3D));
G=8;
for g=1:G
    yin(:,:,:,g)=convn(Im_maps_3D(:,:,:,g),ones(3,3,3)/27,'same');
end

% estimate motion vetors between two consecutive phases (cyclic way)
M1=zeros(26,29,17,3,G);
for i=1:G-1
    [vx,vy,vz]=motionele3d(yin(:,:,:,i:i+1),50,1); % optimized lambda
    M1(:,:,:,:,i)=cat(4,vx,vy,vz);
end
[vx,vy,vz]=motionele3d(yin(:,:,:,[G 1]),50,1);
M1(:,:,:,:,G)=cat(4,vx,vy,vz);

% Define output directory 
output_dir = 'motion_vector_sbta_1e-4_8e6_lmda_01';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);  
end

% Visualize motion vectors on each phase
[X, Y] = meshgrid(1:29, 1:26);  % Assuming 26x29 grid for motion vectors
for i = 1:G
    fig = figure('Visible','off');
    slice_index = 11; 
    imagesc(Im_maps_3D(:,:,slice_index,i));  
    colorbar; colormap jet;
    % recon slice (slice index 36 corresponds to 11 in cropped volume)
    title(['Motion flow on noisy transverse slice 36 for Cardiac Phase-', num2str(i)]);
    hold on; 

    % Extract motion vectors for the current phase
    vx = M1(:,:,slice_index,1,i);  
    vy = M1(:,:,slice_index,2,i); 

    % overlay motion vectors
    quiver(X, Y, vx, vy, 2, 'w'); % scale=2, color=white
    hold off; 

    % Remove Axes Toolbar 
    ax = gca;  % Get current axes
    ax.Toolbar = [];  % Remove toolbar

    % Save the figure
    filename = sprintf('%s/motion_vector_phase_%d.png', output_dir, i);
    saveas(fig, filename);
    close(fig);
end






% Optimizing lambda parameters (SSIM and RMSE) by estimating cardiac motion and motion matrix
load("Im_maps_truth_8e6.mat");
x = Im_maps_truth(:,:,:,6); 

yin=zeros(size(Im_maps_3D));
G=8;
for g=1:G
    yin(:,:,:,g)=convn(Im_maps_3D(:,:,:,g),ones(3,3,3)/27,'same');
end

% Lambda range: low -> more details & noise; high -> more smoothing & blur
lambda_range = [0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.7,1,3,7,10]; 
rmse_values = zeros(size(lambda_range));
ssim_values = zeros(size(lambda_range));
best_rmse = Inf;
best_ssim = -Inf; 
best_lambda_rmse = NaN;
best_lambda_ssim = NaN;

output_dir = 'Optimiz_lambda_sbta_1e-4_8e6';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);  
end

for i = 1:length(lambda_range)
    lambda = lambda_range(i);
    fprintf('Testing lambda = %.2f (%d/%d)\n', lambda, i, length(lambda_range));

    % Estimate cardiac motion with current lambda
    M1 = zeros(26, 29, 17, 3, G);
    for j = 1:G-1
        [vx, vy, vz] = motionele3d(yin(:,:,:,j:j+1), 50, lambda);
        M1(:,:,:,:,j) = cat(4, vx, vy, vz);
    end
    [vx, vy, vz] = motionele3d(yin(:,:,:,[G 1]), 50, lambda);
    M1(:,:,:,:,G) = cat(4, vx, vy, vz);

    % Generate neighboring gate indices
    ind5 = zeros(G, 5);
    for g=1:G
        ind5(g,:) = mod((g-2:g+2), G);
    end
    ind5(ind5 == 0) = G;

    % Generate motion compensation matrices
    for gate = 1:G
        disp(['Processing Gate ', num2str(gate)]);
        MM = mf2matrix3d_5g(M1(:,:,:,:,ind5(gate,:)), 1);
        filename = ['n4dMM' num2str(gate)];
        save(filename, 'MM');
    end

    % Load motion compensation matrix for phase 6
    data = load('n4dMM6.mat');
    MM_gate6 = data.MM;
    disp(size(MM_gate6));

    % Image data set for multiplication
    y1 = Im_maps_3D(:,:,:,4);
    y1_flat = y1(:);
    y2 = Im_maps_3D(:,:,:,5);
    y2_flat = y2(:);
    y3 = Im_maps_3D(:,:,:,7);
    y3_flat = y3(:);
    y4 = Im_maps_3D(:,:,:,8);
    y4_flat = y4(:);

    % Concatenate all vectors into a single column vector
    y = [y1_flat; y2_flat; y3_flat; y4_flat];
    y = double(y);

    % Size check before multiplication
    if size(MM_gate6, 2) ~= size(y, 1)
        error('Matrix size mismatch: MM_gate6 has %d columns but y has %d rows', ...
              size(MM_gate6, 2), size(y, 1));
    end

    % Apply motion compensation
    y = MM_gate6 * y;

    % Reshape back to original 3D image size
    y = reshape(y, 26, 29, 17);
    y = single(y);

    % Display and save the reconstructed image
    fig = figure('Visible','off');
    imagesc(y(:,:,11));  
    colorbar; colormap jet; 
    title(sprintf('Motion Compansated Transverse Slice 36 - Phase 6 (lambda = %.2f)', lambda));
    ax = gca;  % Get current axes
    ax.Toolbar = [];  % Remove toolbar
    filename = sprintf('%s/motion_test_lambda_%.2f.png', output_dir, lambda);
    saveas(fig, filename);
    close(fig);

    % Compute RMSE
    rmse_values(i) = sqrt(mean((x(:) - y(:)).^2));

    % Compute SSIM
    ssim_values(i) = ssim(y, x);

    % Update best RMSE lambda
    if rmse_values(i) < best_rmse
        best_rmse = rmse_values(i);
        best_lambda_rmse = lambda;
    end

    % Update best SSIM lambda
    if ssim_values(i) > best_ssim
        best_ssim = ssim_values(i);
        best_lambda_ssim = lambda;
    end

    close(gcf);
end

% Display best results
fprintf('\nBest lambda (RMSE): %.2f with RMSE = %.2f\n', best_lambda_rmse, best_rmse);
fprintf('Best lambda (SSIM): %.2f with SSIM = %.2f\n', best_lambda_ssim, best_ssim);

% Plot RMSE and SSIM vs lambda 
figure;
yyaxis left  
plot(lambda_range, rmse_values, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'RMSE'); 
ylabel('RMSE');
hold on;
plot(best_lambda_rmse, best_rmse, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10, 'DisplayName', 'Best RMSE');

yyaxis right  
plot(lambda_range, ssim_values, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SSIM');
ylabel('SSIM');
plot(best_lambda_ssim, best_ssim, 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 10, 'DisplayName', 'Best SSIM');

% Labels and Titles
xlabel('Lambda');
title('RMSE and SSIM vs. Lambda');

% Add best values as text on the plot
yyaxis left
text(best_lambda_rmse, best_rmse, sprintf('%.2f, %.2f', best_lambda_rmse, best_rmse), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'FontWeight', 'bold');

yyaxis right
text(best_lambda_ssim, best_ssim, sprintf('%.2f, %.2f', best_lambda_ssim, best_ssim), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 10, 'FontWeight', 'bold');

% Legend
legend('Location', 'best');

% Grid for better visualization
grid on;
hold off;

% Save the figure
saveas(gcf, sprintf('%s/RMSE_SSIM_vs_lambda.png', output_dir));
close(gcf);




% Optimizing sbeta values in 3d using both RMSE and SSIM
gbeta = 0; % Temporal regularization weight (0 = no temporal smoothing)
sub_num = 16; % Reduce for noisy images, increase for stability
it_num = 10;  % Increase for better recon

sbeta_values = [1e-5,5e-5,7e-5,1e-4,5e-4,7e-4,1e-3,5e-3,1e-2]; 
rmse_values = zeros(size(sbeta_values)); 
ssim_values = zeros(size(sbeta_values)); 
best_rmse = Inf;  
best_ssim = -Inf;  
best_sbeta_rmse = NaN;
best_sbeta_ssim = NaN;

% Load ground truth
load('Im_maps_truth_8e6.mat');

% Define output directory
output_dir = 'Optimiz_sbeta_3d_8e6';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);  
end

for i = 1:length(sbeta_values)
    sbeta = sbeta_values(i);
    fprintf('Testing sbeta = %.1e (%d/%d)\n', sbeta, i, length(sbeta_values));

    tic;
    Im_maps = mbsrem4dv2(projections, repmat(roi, [1,1,64,8]), ...
        sub_num, it_num, OF_tag, sbeta, gbeta, blur, 1, 0, 0);
    toc;

    Im_maps(Im_maps < 0) = 0;
    Im_maps = Im_maps / sum(Im_maps(:)) * total_counts;

    % Define ROI (heart region)
    minx=16; maxx=41;
    miny=15; maxy=43;
    minz=26; maxz=42;

    % Extract heart region
    Im_maps_actual = Im_maps(minx:maxx, miny:maxy, minz:maxz, :); 

    % Save the recon slice (slice index 36 corresponds to 11 in cropped volume)
    fig = figure('Visible', 'off');  
    imagesc(Im_maps_actual(:,:,11,1));  
    colorbar; 
    colormap jet;
    title(['Noisy 3D recon - transverse slice 36 for phase 1 (sbeta = ', num2str(sbeta), ')']);
    ax = gca;  % get current axis  
    ax.Toolbar = [];  % remove toolbar 
    filename = sprintf('%s/recon_sbeta_%.1e.png', output_dir, sbeta);
    saveas(fig, filename);
    close(fig);  

    % Compute RMSE
    rmse_values(i) = sqrt(mean((Im_maps_actual(:,:,:,1) - Im_maps_truth(:,:,:,1)).^2, 'all'));

    % Compute SSIM 
    ssim_values(i) = ssim(Im_maps_actual(:,:,:,1), Im_maps_truth(:,:,:,1));

    % Find best RMSE
    if rmse_values(i) < best_rmse
        best_rmse = rmse_values(i);
        best_sbeta_rmse = sbeta;
    end

    % Find best SSIM
    if ssim_values(i) > best_ssim
        best_ssim = ssim_values(i);
        best_sbeta_ssim = sbeta;
    end

    close(gcf);
end

% Display best results
fprintf('\nBest RMSE: %.2f at sbeta = %.1e\n', best_rmse, best_sbeta_rmse);
fprintf('Best SSIM: %.2f at sbeta = %.1e\n', best_ssim, best_sbeta_ssim);

% Plot RMSE and SSIM vs sbeta 
figure;
yyaxis left  
plot(sbeta_values, rmse_values, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'RMSE'); 
ylabel('RMSE');  
hold on;
plot(best_sbeta_rmse, best_rmse, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10, 'DisplayName', 'Best RMSE');

yyaxis right  
plot(sbeta_values, ssim_values, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SSIM');
ylabel('SSIM');
plot(best_sbeta_ssim, best_ssim, 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 10, 'DisplayName', 'Best SSIM');

% Labels and Titles
xlabel('sbeta');
title('RMSE and SSIM vs. sbeta');

% Annotate best RMSE and SSIM values
yyaxis left
text(best_sbeta_rmse, best_rmse, sprintf('%.1e, %.2f', best_sbeta_rmse, best_rmse), ...
'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'r');

yyaxis right
text(best_sbeta_ssim, best_ssim, sprintf('%.1e, %.2f', best_sbeta_ssim, best_ssim), ...
'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'b');

% Legend
legend('Location', 'best');

% Grid for better visualization
grid on;
hold off;

% Save the figure
saveas(gcf, sprintf('%s/RMSE_SSIM_vs_sbeta.png', output_dir));
close(gcf);





