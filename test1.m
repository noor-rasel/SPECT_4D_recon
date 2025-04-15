%% read SIMIND projection data
projections = zeros(64,64,64,8);
total_counts = 8e6; 
total_counts_per_resp_phase = 8e6/8; 
blur = 1; % include the distance-dependent blur (PSF modelling)
OF_tag=0; % calculate MAP objective function, zero means "No"
load('roi.mat'); 
rng(20230110);
for resp = 0 : 0  % just for resp phase #1
    for cardica = 0 : 7
        filename = sprintf('cardiac%d_air_w1.a00',  cardica+1);
        fid = fopen(filename,'rb');
        simind = fread(fid,'float');
        fclose(fid);
        simind = reshape(simind,64,64,64);
        projections(:,:,:,cardica+1) = simind;

    end
end
disp(size(projections));

%% Proper orientation of projection matrix and normalization   
for i = 1:size(projections, 4)
    for j = 1:size(projections, 3) 
        projections(:,:,j,i) = rot90(projections(:,:,j,i), 3);
    end
end

disp(['Counts before normalization: ', num2str(sum(projections(:)))]);

projections = projections / sum(projections(:)) * total_counts;
projections(projections<0) = 0;
disp(['Counts after normalization: ', num2str(sum(projections(:)))]);

%  % Display multiple slices for a fixed cardiac phase
% for n = 26:42
%     figure;
%     imagesc(projections(:,:,n,1));  
%     colorbar; colormap jet; % jet, hot, turbo, parula
%     title(['Transverse Slice ', num2str(n), ' - Cardiac Phase 1']);
%     pause;
% end

%% Ideal 3d recon 
gbeta=0; % temproal regulization weight (0, no temproal smoothing)
sbeta=0; % gbeta=0 & sbeta=0 <=> OSEM
sub_num=16; % or 8, [1,4,8,16]
it_num=10; % or 40, [10,20,30,40,50], decrease sub_num, increase it_num to avoid excessive noise and full convergence
tic; 
Im_maps=mbsrem4dv2(projections,repmat(roi,[1,1,64,8]),sub_num,it_num,OF_tag,...
    sbeta,gbeta,blur,0,0,0); 
toc;
disp(['Image data counts before norm: ', num2str(sum(Im_maps(:)))]);

Im_maps=Im_maps/sum(Im_maps(:)) * total_counts;
Im_maps(Im_maps<0)=0;
disp(['Image data counts after norm: ', num2str(sum(Im_maps(:)))]);

% Global coordinates (minx:maxx, miny:maxy, minz:maxz) defines the ROI (LV/heart region) inside the full 64×64×64 space. 
minx=16;maxx=41;
miny=15;maxy=43;
minz=26;maxz=42;

Im_maps_truth=Im_maps(minx:maxx,miny:maxy,minz:maxz,:);
disp(size(Im_maps_truth)); 
save('Im_maps_truth_8e6.mat',"Im_maps_truth");

output_dir = '3D_ideal_recon_images_8e6'; 
if ~exist(output_dir, 'dir')
    mkdir(output_dir); 
end

% Save the recon slice (slice index 36 corresponds to 11 in cropped volume)
for n = 1:8
    fig = figure('Visible', 'off'); 
    imagesc(Im_maps_truth(:,:,11,n));  
    colorbar; 
    colormap jet; 
    title(['Ideal 3D recon transverse slice 36 for cardiac phase-', num2str(n)]);
    filename = sprintf('%s/recon_phase_%d.png', output_dir, n);
    saveas(fig, filename);
    close(fig);  
end

