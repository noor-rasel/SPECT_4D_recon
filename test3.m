%% read SIMIND projection data
path_setting;
data_root_folder = simind_attn_data_root_folder + "all_phases";
addpath(recon_root_folder+"code/");
addpath(recon_root_folder+"code/4D SPECT recon with ASC/4D/");
addpath(recon_root_folder+"code/4D SPECT recon with ASC/weight_gen/");
addpath(recon_root_folder+"code/extra_files/",'-begin');

projections_ori = zeros(64,64,64,8);
total_projections = zeros(64,64,64,8,8);
total_counts = 8e6; 
total_counts_cp1_rp1 = 8e6/64;
blur = 1; % include the distance-dependent blur (PSF modelling)
OF_tag=0; % calculate MAP objective function, zero means "No"
load('roi.mat'); 
rng(20230110);

for resp = 0 : 7
    for cardica = 0 : 7
        filename = sprintf('%s/cardica%d_resp_%d/cardiac%d_tot_w1.a00', data_root_folder, cardica+1, resp+1, cardica+1);
        fid = fopen(filename,'rb');
        simind = fread(fid,'float');
        fclose(fid);
        simind = reshape(simind,64,64,64);        
        projections_ori(:,:,:,cardica+1) = simind;
    end

    %% set projection properly and add noise
    projections = projections_ori;
    for i = 1:size(projections_ori,4)
        cur_proj = projections_ori(:,:,:,i);
        for j = 1:64
            cur_proj(:,:,j) = rot90(cur_proj(:,:,j),3);
        end
        projections(:,:,:,i) = cur_proj;
    end
    one_projection = projections(:,:,:,1);
    disp(size(one_projection));
    count_one_projection = sum(one_projection(:));
    disp(['Counts for one card phase : ', num2str(sum(count_one_projection(:)))]);
    projections = projections/ count_one_projection * total_counts_cp1_rp1;
    disp(['Counts for 8 card phases : ', num2str(sum(projections(:)))]);
    projections = random('poiss', projections);
    total_projections(:,:,:,:,resp+1) = projections;
    disp(size(total_projection));
    disp(['Total counts: ', num2str(sum(totalprojections(:)))]);
end
