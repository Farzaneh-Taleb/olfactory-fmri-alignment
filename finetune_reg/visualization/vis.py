import os , sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
project_root = '/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment-NEW'
sys.path.insert(0, project_root)
from utils.vis_helper import *
# ----------------- Usage -----------------
base_dir = "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment"
out_dir = f"{base_dir}/final-figs"


start ='Sep20'
end = '11_finetune_by_fmri'
#fMRI
df_fmri = load_metrics_fast(f"{base_dir}/Sep19_fmrimetrics_5_alphapertarget")
df_fmri['type'] = 'frozen'
df_fmri["_origin"] = "behavior:frozen"
print("df_fmri:", df_fmri.columns.values.tolist(), df_fmri.shape)

df_fmrituned = load_metrics_fast(f"{base_dir}/{start}_fmrifinetuned_metrics_{end}")
df_fmrituned['type'] = 'fine-tuned'
df_fmrituned["_origin"] = "behavior:fine-tuned"
print("df_fmrituned:", df_fmrituned.columns.values.tolist(), df_fmrituned.shape)

df_fmri_bars   = add_bar_group_column(df_fmri, type_col="type", source_col="participant_source_id", out_col="bar_group")
df_tuned_bars  = add_bar_group_column(df_fmrituned, type_col="type", source_col="participant_source_id", out_col="bar_group")



p_vals =[1, 0.05]


#all the plots that separted by tr and roi
# participant_id * model
for p_val in p_vals:
    for unfreeze_last_n in df_tuned_bars["unfreeze_last_n"].unique():
        for n_components in df_tuned_bars["n_components"].unique():
                for tr in df_tuned_bars["tr"].unique():
                    for roi in df_tuned_bars["roi"].unique():
                        print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, n_components={n_components}, p_val={p_val} ===")

                        df_fmri_filtered = df_filter(
                            df_fmri_bars,
                            filters={
                                "n_components": n_components,
                                "tr": tr,
                                "roi": roi,

                                # NOTE: no tr/roi here – we want *all* positive TRs and all ROIs in one figure
                            })
                        df_fmrituned_filtered = df_filter(
                            df_tuned_bars,
                            filters={
                                "unfreeze_last_n": unfreeze_last_n,
                                "n_components": n_components,
                                "tr": tr,
                                "roi": roi,

                            })
                        df_fmri_last_bars  = filter_last_layer(df_fmri_filtered)
                        df_fmrituned_last_bars = filter_last_layer(df_fmrituned_filtered)
                        df_combo_last = pd.concat([df_fmri_last_bars, df_fmrituned_last_bars], ignore_index=True)

                        #1
                        plot_avg_over_descriptors_by_participant_grid(
                            df_combo_last,
                            value_col="correlation",
                            target_col="target",
                            pid_col="participant_id",
                            model_col="model",
                            type_col="type",
                            bar_group_col="bar_group",
                            pval_col="p_value_correlation",
                            pval_thresh=p_val,
                            agg="mean",
                            error="sem",
                            min_targets=3,
                            save_dir=out_dir,
                            filename_stub=f"avg_desc/fmri_avg_over_desc_last_unf-{slug(unfreeze_last_n)}-_ncomp-{slug(n_components)}tr-{slug(tr)}__roi-{slug(roi)}_{start}_{end}",
                            show=False,
                        )
                    


# start = 'Sep19'
# end = '5'


#behavior
# df_behavior = _load_metrics_fast(f"{base_dir}/{start}_behaviormetrics_{end}_alphapertarget")
# df_behavior['type'] = 'frozen'
# df_behavior["_origin"] = "behavior:frozen"
# print("df_fmri:", df_behavior.columns.values.tolist(), df_behavior.shape)

# df_behaviortuned = _load_metrics_fast(f"{base_dir}/{start}_behaviortuned_metrics_{end}_alphapertarget")
# df_behaviortuned['type'] = 'fine-tuned'
# df_behaviortuned["_origin"] = "behavior:fine-tuned"
# print("df_fmrituned:", df_behaviortuned.columns.values.tolist(), df_behaviortuned.shape)

# df_behavior_bars   = add_bar_group_column(df_behavior, type_col="type", source_col="participant_source_id", out_col="bar_group")
# df_behaviortuned_bars  = add_bar_group_column(df_behaviortuned, type_col="type", source_col="participant_source_id", out_col="bar_group")
# df_behavior_last_bars  = filter_last_layer(df_behavior_bars)
# df_behaviortuned_last_bars = filter_last_layer(df_behaviortuned_bars)

# df_behavior_max_bars = reduce_to_max_layer_per_target(
#             df_behavior_bars,
#             group_cols=("participant_id", "model", "type", "target","n_components"),
#             corr_col="correlation",
#             layer_col="layer",
#             pval_col="p_value_correlation"  # if present
           
# )
# df_behaviortuned_max_bars = reduce_to_max_layer_per_target(
#             df_behaviortuned_bars,
#             group_cols=("participant_id", "model", "type", "target","n_components","unfreeze_last_n"),
#             corr_col="correlation",
#             layer_col="layer",
#             pval_col="p_value_correlation"  # if present
# )

# p_vals =[1, 0.05]

# for p_val in p_vals:
#     for unfreeze_last_n in df_behaviortuned_max_bars["unfreeze_last_n"].unique():
#         # for behavior_embeddings in df_tuned_bars["behavior_embeddings"].unique():
#         for n_components in df_behaviortuned_max_bars["n_components"].unique():
#             # for model in df_behaviortuned_max_bars["model"].unique():
                
#                     print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, n_components={n_components}, p_val={p_val} ===")

#                     df_behavior_filtered = df_filter(
#                         df_behavior_bars,
#                         filters={
#                             "n_components": n_components,
#                             # "model": model,
                           

#                             # NOTE: no tr/roi here – we want *all* positive TRs and all ROIs in one figure
#                         })
#                     df_behaviortuned_filtered = df_filter(
#                         df_behaviortuned_bars,
#                         filters={
#                             "unfreeze_last_n": unfreeze_last_n,
#                             "n_components": n_components,
#                             # "model": model,
                            

#                         })
#                     # df_combo_layers = pd.concat([df_behavior_filtered, df_behavioruned_filtered], ignore_index=True)
#                     df_all = pd.concat([df_behaviortuned_filtered, df_behavior_filtered], ignore_index=True)
#                     print("df_all:", df_all.columns.values.tolist(), df_all.shape)
#                     df_all_last = filter_last_layer(df_all.copy())
#                     print("df_all_last:", df_all.columns.values.tolist(), df_all_last.shape)
#                     df_all_max = reduce_to_max_layer_per_target(
#                     df_all,
#                     group_cols=("participant_id", "model", "type", "target"),
#                     corr_col="correlation",
#                     layer_col="layer",
#                     pval_col="p_value_correlation"  # if present
#                     , pval_thresh=p_val  # if present
#         )
#                                 # build a filename stub from the selections
#                     # beh_stub = _slug(behavior_embeddings)
#                     stub = f"ds-sagar2023__unf-{_slug(unfreeze_last_n)}__ncomp-{_slug(n_components)}_{start}_{end}__pval-{_slug(p_val)}"
#                     print(df_all_last['target'].unique())
#                     # save one PNG per participant automatically
#                     plot_correlation_bars_by_participant_grid(
#                         df_all_last,
#                         corr_col="correlation",
#                         target_col="target",
#                         pid_col="participant_id",
#                         type_col="type",
#                         pval_col="p_value_correlation",
#                         pval_thresh=p_val,
#                         sort_targets="mean",
#                         save_dir=out_dir,
#                         filename_stub=stub,
#                         show=False
#                     )
#                     plot_maxcorr_bars_by_participant_x_model(
#                     df_all_max,
#                     corr_col="correlation",
#                     target_col="target",
#                     pid_col="participant_id",
#                     model_col="model",
#                     type_col="type",
#                     layer_at_max_col="layer",
#                     pval_col="p_value_correlation",      # threshold on pval from the max layer
#                     pval_thresh=p_val,
#                     sort_targets="mean",
#                     annotate_layer=True,
#                     save_dir=out_dir,
#                     filename_stub=stub + "__maxlayerrrr",
#                     show=False
#                         )
                                                                                                        #   for unfreeze_last_n in df_behaviortuned["unfreeze_last_n"].unique():
        #     for behavior_embeddings in df_behaviortuned["behavior_embeddings"].unique():
        #         for n_components in df_behavior["n_components"].unique():
        #             df_behaviortuned_filtered = df_filter(
        #                 df_behaviortuned,
        #                 filters={
        #                     "ds": "sagar2023",
        #                     "unfreeze_last_n": unfreeze_last_n,
        #                     "behavior_embeddings": behavior_embeddings,
        #                     "n_components": n_components,
        #                 })
        #             df_behavior_filtered = df_filter(
        #                 df_behavior,
        #                 filters={"ds": "sagar2023"}
        #             )
        #                    # combine, keep last layer
        #             df_all = pd.concat([df_behaviortuned_filtered, df_behavior_filtered], ignore_index=True)
        #             print("df_all:", df_all.columns.values.tolist(), df_all.shape)
        #             df_all_last = filter_last_layer(df_all.copy())
        #             print("df_all_last:", df_all.columns.values.tolist(), df_all_last.shape)
        #             df_all_max = reduce_to_max_layer_per_target(
        #             df_all,
        #             group_cols=("participant_id", "model", "type", "target"),
        #             corr_col="correlation",
        #             layer_col="layer",
        #             pval_col="p_value_correlation"  # if present
        #             , pval_thresh=0.05  # if present
        # )
        #                         # build a filename stub from the selections
        #             beh_stub = _slug(behavior_embeddings)
        #             stub = f"ds-sagar2023__unf-{_slug(unfreeze_last_n)}__beh-{beh_stub}__ncomp-{_slug(n_components)}"
        #             print(df_all_last['target'].unique())
        #             # save one PNG per participant automatically
        #             plot_correlation_bars_by_participant_grid(
        #                 df_all_last,
        #                 corr_col="correlation",
        #                 target_col="target",
        #                 pid_col="participant_id",
        #                 type_col="type",
        #                 pval_col="p_value_correlation",
        #                 pval_thresh=0.05,
        #                 sort_targets="mean",
        #                 save_dir=out_dir,
        #                 filename_stub=stub,
        #                 show=False
        #             )
        #             plot_maxcorr_bars_by_participant_x_model(
        #             df_all_max,
        #             corr_col="correlation",
        #             target_col="target",
        #             pid_col="participant_id",
        #             model_col="model",
        #             type_col="type",
        #             layer_at_max_col="layer",
        #             pval_col="p_value_correlation",      # threshold on pval from the max layer
        #             pval_thresh=0.05,
        #             sort_targets="mean",
        #             annotate_layer=True,
        #             save_dir=out_dir,
        #             filename_stub=stub + "__maxlayerrrr",
        #             show=False
        #                 )






# issues_behavior,df_behavior_clean = audit_metrics(df_behavior, name="df_behavior", tuned=False)
# issues_beh_tuned,df_behaviortuned_clean = audit_metrics(df_behaviortuned, name="df_behaviortuned", tuned=True)

# for tag, bad in issues_behavior.items():
#     print(f"\n[behavior] {tag}")
#     print(bad[["source_file","row_idx"]].head())

# for tag, bad in issues_beh_tuned.items():
#     print(f"\n[behaviortuned] {tag}s")
#     print(bad[["source_file","row_idx"]].head())

# for unfreeze_last_n in df_behaviortuned_clean["unfreeze_last_n"].unique():
#     for behavior_embeddings in df_behaviortuned_clean["behavior_embeddings"].unique():
#         for n_components in df_behavior_clean["n_components"].unique():
#             # print(f"behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
#             print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
#             df_behaviortuned_filtered = df_filter(
#                 df_behaviortuned_clean,
#                 filters={
#                     "ds": "sagar2023",
#                     # "unfreeze_last_n": unfreeze_last_n,
#                     "behavior_embeddings": behavior_embeddings,
#                     "n_components": n_components,
#                 })
#             df_behavior_filtered = df_filter(
#                 df_behavior_clean,
#                 filters={"ds": "sagar2023"}
#             )
#             print("df_behaviortuned_filtered:", df_behaviortuned_filtered.columns.values.tolist(), df_behaviortuned_filtered.shape)

#             # combine, keep last layer
#             df_all = pd.concat([df_behaviortuned_filtered, df_behavior_filtered], ignore_index=True)
#             print("df_all:", df_all.columns.values.tolist(), df_all.shape)
#             df_all_last = filter_last_layer(df_all.copy())
#             print("df_all_last:", df_all.columns.values.tolist(), df_all_last.shape)
#             df_all_max = reduce_to_max_layer_per_target(
#             df_all,
#             group_cols=("participant_id", "model", "type", "target"),
#             corr_col="correlation",
#             layer_col="layer",
#             pval_col="p_value_correlation"  # if present
#             , pval_thresh=p_val  # if present
# )


#             # build a filename stub from the selections
#             beh_stub = _slug(behavior_embeddings)
#             stub = f"ds-sagar2023__unf-{_slug(unfreeze_last_n)}__beh-{beh_stub}__ncomp-{_slug(n_components)}_{start}_{end}"
#             print(df_all_last['target'].unique())






            # save one PNG per participant automatically
            # plot_correlation_bars_by_participant_grid(
            #     df_all_last,
            #     corr_col="correlation",
            #     target_col="target",
            #     pid_col="participant_id",
            #     type_col="type",
            #     pval_col="p_value_correlation",
            #     pval_thresh=0.05,
            #     sort_targets="mean",
            #     save_dir=out_dir,
            #     filename_stub=stub,
            #     show=False
            # )
            # plot_maxcorr_bars_by_participant_x_model(
            # df_all_max,
            # corr_col="correlation",
            # target_col="target",
            # pid_col="participant_id",
            # model_col="model",
            # type_col="type",
            # layer_at_max_col="layer",
            # pval_col="p_value_correlation",      # threshold on pval from the max layer
            # pval_thresh=0.05,
            # sort_targets="mean",
            # annotate_layer=True,
            # save_dir=out_dir,
            # filename_stub=stub + "_maxlayer",
            # show=False
            #     )
            

            # plot_avg_over_descriptors_by_participant_grid(
            # df_all_last,
            # value_col="correlation",
            # targe#t_col="target",
            # pid_col="participant_id",
            # model_col="model",
            # type_col="type",
            # pval_col="p_value_correlation",
            # pval_thresh=0.05,      # optional
            # agg="mean",            # or "median"
            # error="sem",           # "std" or None
            # min_targets=3,
            # save_dir=out_dir,
            # filename_stub=stub + f"__avg_over_desc_last_{p_val}",
            # show=False
            # )       
#             fisher_z = True      
#             plot_avg_over_descriptors_by_participant_grid(
#     df_all_max,
#     value_col="correlation",
#     pid_col="participant_id",
#     model_col="model",
#     type_col="type",
#     target_col="target",
#     compare_types=("frozen","fine-tuned"),  # BEFORE, AFTER
#     test_kind="wilcoxon",                       # or "ttest"
#     fisher_z=fisher_z,
#     show=False,
#     filename_stub=stub + f"__avg_over_desc_max_p_{p_val}_{fisher_z}",
#     save_dir=out_dir,
#     error="sem",
#      pval_thresh=p_val,
     
# )
            

#             fisher_z = True      
#             plot_avg_over_descriptors_by_participant_grid(
#     df_all_max,
#     value_col="correlation",
#     pid_col="participant_id",
#     model_col="model",
#     type_col="type",
#     target_col="target",
#     compare_types=("frozen","fine-tuned"),  # BEFORE, AFTER
#     test_kind="wilcoxon",                       # or "ttest"
#     fisher_z=fisher_z,
#     show=False,
#     filename_stub=stub + f"__avg_over_desc_max_p_{p_val}_{fisher_z}",
#     save_dir=out_dir,
#     error="sem",
#      pval_thresh=p_val,
     
# )


            # plot_avg_over_descriptors_by_participant_grid(
            # df_all_max,
            # value_col="correlation",
            # target_col="target",
            # pid_col="participant_id",
            # model_col="model",
            # type_col="type",
            # agg="mean",
            # error="sem",
            # min_targets=3,
            # save_dir=out_dir,
            # filename_stub=stub + f"__avg_over_desc_max_{p_val}",
            # show=False,
           
            # )  





# df_fmri = load_all_metrics(f"{base_dir}/Sep16_fmrimetrics_1")
# df_fmri['type'] = 'frozen'
# print("df_fmri:", df_fmri.columns.values.tolist(), df_fmri.shape)


# df_fmri_tuned = load_tuned_metrics2(f"{base_dir}/Sep16_fmrifinetuned_metrics_1")
# df_fmri_tuned['type'] = 'fine-tuned'
# print("df_fmri_tuned:", df_fmri_tuned.columns.values.tolist(), df_fmri.shape)
# issues_behavior,df_behavior_clean = audit_metrics(df_fmri, name="df_fmri", tuned=False)
# issues_beh_tuned,df_behaviortuned_clean = audit_metrics(df_fmri_tuned, name="df_fmrituned", tuned=True)

# <<< NEW: load new-style CSVs that include participant_source_id (adjust directory as needed)
# Example usage if you place them under a folder like below:
# df_sources = load_any_metrics_in_dir(f"{base_dir}/Sep16_fmri_sources")
# If you don't have such a folder yet, comment out the next two lines.








                    # plot_fmri_layer_lines_trpairs_by_roi_subject(
                    #     df_combo_layers,
                    #     value_col="correlation",
                    #     target_col="target",
                    #     pid_col="participant_id",
                    #     roi_col="roi",
                    #     tr_col="tr",
                    #     type_col="type",             # expects 'frozen' and 'fine-tuned'
                    #     model_col="model",
                    #     pval_col="p_value_correlation",
                    #     pval_thresh=p_val,            # or 1 for no filtering
                    #     fisher_z=False,              # set True if you want z before averaging
                    #     min_targets=3,
                    #     sharey=True,
                    #     save_dir=out_dir,
                    #     filename_stub=f"fmri_layer_trpairs__model-ALL__minT3_{start}_{end}_{n_components}ncomp__unf-{slug(unfreeze_last_n)}_{p_val}_{model}_tr-{tr}",
                    #     show=False,
                    # )

# df_fmri_max_bars = reduce_to_max_layer_per_target(
#             df_fmri_bars,
#             group_cols=("participant_id", "model", "type", "target","tr","roi","n_components"),
#             corr_col="correlation",
#             layer_col="layer",
#             pval_col="p_value_correlation"  # if present
           
# )
# df_tuned_max_bars = reduce_to_max_layer_per_target(
#             df_tuned_bars,
#             group_cols=("participant_id", "model", "type", "target","tr","roi","n_components","unfreeze_last_n"),
#             corr_col="correlation",
#             layer_col="layer",
#             pval_col="p_value_correlation"  # if present
# )
# for p_val in p_vals:
#     for unfreeze_last_n in df_tuned_bars["unfreeze_last_n"].unique():
#         # for behavior_embeddings in df_tuned_bars["behavior_embeddings"].unique():
#         for n_components in df_tuned_bars["n_components"].unique():
#             for model in df_tuned_bars["model"].unique():
#                 print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, n_components={n_components}, model={model}, p_val={p_val} ===")
    
#                 df_fmri_filtered = df_filter(
#                     df_fmri_max_bars,
#                     filters={
#                         "n_components": n_components,
#                         "model": model,
#                     })
#                 df_fmrituned_filtered = df_filter(
#                     df_tuned_max_bars,
#                     filters={
#                         "unfreeze_last_n": unfreeze_last_n,
#                         "n_components": n_components,
#                         "model": model,
#                     })
#                 df_combo_layers = pd.concat([df_fmri_filtered, df_fmrituned_filtered], ignore_index=True)
#                 plot_maxavg_over_layer_tr_by_model_grid(
#                     df_combo_layers,
#                     value_col="correlation",
#                     target_col="target",
#                     pid_col="participant_id",
#                     roi_col="roi",
#                     tr_col="tr",
#                     type_col="type",             # expects 'frozen' and 'fine-tuned'
#                     model_col="model",
#                     pval_col="p_value_correlation",
#                     pval_thresh=p_val,            # or 1 for no filtering
#                     fisher_z=False,              # set True if you want z before averaging
#                     min_targets=3,
#                     sharey=True,
#                     save_dir=out_dir,
#                     filename_stub=f"fmri_max_over_LxTR_avg_by_model_{start}_{end}_{n_components}ncomp__unf-{_slug(unfreeze_last_n)}_{p_val}_{model}",
#                     show=False,
#                 )
                # plot_max_travg_over_layers_by_model_grid(
                #     df_combo_layers,
                #     value_col="correlation",
                #     target_col="target",
                #     pid_col="participant_id",
                #     roi_col="roi",
                #     tr_col="tr",
                #     type_col="type",             # expects 'frozen' and 'fine-tuned'
                #     model_col="model",                    pval_col="p_value_correlation",
                #     pval_thresh=p_val,            # or 1 for no filtering
                #     fisher_z=False,              # set True if you want z before averaging
                #     min_targets=3,
                #     sharey=True,
                #     save_dir=out_dir,
                #     filename_stub=f"fmri_max_over_LayerxTR_avg_by_model_{start}_{end}_{n_components}ncomp__unf-{_slug(unfreeze_last_n)}_{p_val}_{model}",
                #     show=False,
                # )




# for unfreeze_last_n in df_tuned_bars["unfreeze_last_n"].unique():
#     for behavior_embeddings in df_tuned_bars["behavior_embeddings"].unique():
#         for n_components in df_tuned_bars["n_components"].unique():
#             for tr in df_tuned_bars["tr"].unique():
#                 for roi in df_tuned_bars["roi"].unique():
                   
#                     print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
#                     df_fmri_filtered = df_filter(
#                         df_fmri_bars,
#                         filters={
#                             # "unfreeze_last_n": unfreeze_last_n,
#                             # "behavior_embeddings": behavior_embeddings,
#                             "n_components": n_components,
#                             "ds": "sagar2023",
#                             "tr": tr,
#                             "roi": roi,
#                         })
#                     print("df_fmri_filtered:", df_fmri_filtered.columns.values.tolist(), df_fmri_filtered.shape)

#                     df_fmrituned_filtered = df_filter(
#                         df_tuned_bars,
#                         filters={
#                             "unfreeze_last_n": unfreeze_last_n,
#                             "behavior_embeddings": behavior_embeddings,
#                             "n_components": n_components,
#                             "ds": "sagar2023",
#                             "tr": tr,
#                             "roi": roi,     
#                         })

#                     print("df_behaviortuned_filtered:", df_fmrituned_filtered.columns.values.tolist(), df_fmri_filtered.shape)

#                     print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, behavior_embeddings={behavior_embeddings}, n_components={n_components}, tr={tr}, roi={roi} ===")
                    
#                     df_fmri_all_combo = pd.concat([df_fmri_filtered, df_fmrituned_filtered], ignore_index=True)
#                     df_fmri_all_last  = filter_last_layer(df_fmri_all_combo)

#                     plot_avg_over_descriptors_by_participant_grid(
#                         df_fmri_all_last,
#                         value_col="correlation",
#                         target_col="target",
#                         pid_col="participant_id",
#                         model_col="model",
#                         type_col="type",
#                         bar_group_col="bar_group",
#                         pval_col="p_value_correlation",
#                         pval_thresh=p_val,
#                         agg="mean",
#                         error="sem",
#                         min_targets=3,
#                         save_dir=out_dir,
#                         filename_stub=f"fmri_avg_over_desc_last__unf-{_slug(unfreeze_last_n)}__beh-{_slug(behavior_embeddings)}__ncomp-{_slug(n_components)}__tr-{_slug(tr)}__roi-{_slug(roi)}_{start}_{end}",
#                         show=False,
#                     )





















            # print(f"behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
   
            # combine, keep last layer
            
# df_sources = load_any_metrics_in_dir(f"{base_dir}/Sep16_behaviortunedtransfer_metrics_1")
# df_sources["_origin"] = "behavior:transfer"
# print("df_sources:", df_sources.columns.values.tolist(), df_sources.shape)

# issues_fmri,df_fmri_clean = audit_metrics_fmri(df_fmri, name="df_fmri", tuned=False)
# issues_fmri_tuned,df_fmrituned_clean = audit_metrics_fmri(df_fmrituned, name="df_fmrituned", tuned=True)
# issues_beh_sources,df_sources_clean = audit_metrics_fmri(df_sources, name="df_sources", tuned=True)

# <<< NEW: if you have df_sources uncommented above, you can combine and add bar_group like this:

# df_sources_bars = add_bar_group_column(df_sources_clean, type_col="type", source_col="participant_source_id", out_col="bar_group")




#Behavior
# df_behavior = load_any_metrics_in_dir(f"{base_dir}/Sep16_behaviormetrics_1")
# df_behavior['type'] = 'frozen'
# df_behavior["_origin"] = "behavior:frozen"
# print("df_behavior:", df_behavior.columns.values.tolist(), df_behavior.shape)

# df_behaviortuned = load_any_metrics_in_dir(f"{base_dir}/{start}_behaviortuned_metrics_{end}")
# df_behaviortuned['type'] = 'fine-tuned'
# df_behaviortuned["_origin"] = "behavior:fine-tuned"
# print("df_behaviortuned:", df_behaviortuned.columns.values.tolist(), df_behaviortuned.shape)

# df_sources = load_any_metrics_in_dir(f"{base_dir}/Sep16_behaviortunedtransfer_metrics_1")
# df_sources["_origin"] = "behavior:transfer"
# print("df_sources:", df_sources.columns.values.tolist(), df_sources.shape)

# issues_behavior,df_behavior_clean = audit_metrics(df_behavior, name="df_behavior", tuned=False)
# issues_beh_tuned,df_behaviortuned_clean = audit_metrics(df_behaviortuned, name="df_behaviortuned", tuned=True)
# issues_beh_sources,df_sources_clean = audit_metrics(df_sources, name="df_sources", tuned=True)

# # <<< NEW: if you have df_sources uncommented above, you can combine and add bar_group like this:
# df_sources_bars = add_bar_group_column(df_sources_clean, type_col="type", source_col="participant_source_id", out_col="bar_group")
# df_behavior_bars   = add_bar_group_column(df_behavior_clean, type_col="type", source_col="participant_source_id", out_col="bar_group")
# df_tuned_bars  = add_bar_group_column(df_behaviortuned_clean, type_col="type", source_col="participant_source_id", out_col="bar_group")



# for tag, bad in issues_behavior.items():
#     print(f"\n[behavior] {tag}")
#     print(bad[["source_file","row_idx"]].head())

# for tag, bad in issues_beh_tuned.items():
#     print(f"\n[behaviortuned] {tag}s")
#     print(bad[["source_file","row_idx"]].head())

# for tag, bad in issues_beh_sources.items():
#     print(f"\n[sources] {tag}s")
#     print(bad[["source_file","row_idx"]].head())


# # # Basic uniqueness checks
# # print("\nUnique run_id by origin:")
# # print(pd.concat([df_behavior, df_behaviortuned, df_sources], ignore_index=True)
# #         .groupby("_origin")["run_id"].nunique())

# # # Do the sources actually vary?
# # if "participant_source_id" in df_sources.columns:
# #     print("\nSources present (participant_source_id):", sorted(pd.to_numeric(
# #         df_sources["participant_source_id"], errors="coerce").dropna().unique().tolist()))

# # # Add bar_group and tag, then check means per category
# # df_behavior_bars = add_bar_group_column(df_behavior, out_col="bar_group")
# # df_tuned_bars    = add_bar_group_column(df_behaviortuned, out_col="bar_group")
# # df_sources_bars  = add_bar_group_column(df_sources, out_col="bar_group")

# # df_all_combo = pd.concat([df_behavior_bars, df_tuned_bars, df_sources_bars], ignore_index=True)
# # df_all_last  = filter_last_layer(df_all_combo)






# # for tag, bad in issues_behavior.items():
# #     print(f"\n[behavior] {tag}")
# #     print(bad[["source_file","row_idx"]].head())

# # for tag, bad in issues_beh_tuned.items():
# #     print(f"\n[behaviortuned] {tag}s")
# #     print(bad[["source_file","row_idx"]].head())




# # print("\nCategories in bar_group:", sorted(df_all_last["bar_group"].dropna().unique().tolist()))

# # # Look at means per (origin vs category) to spot duplicates quickly
# # summary_check = (df_all_last
# #     .groupby(["_origin","bar_group","participant_source_id","participant_id"])["correlation"]
# #     .mean()
# #     .sort_index()
# #     .reset_index())
# # print("\nMean correlations by origin x bar_group:")
# # print(summary_check.head(40))

# # df_all_combo   = pd.concat([df_behavior_bars, df_tuned_bars, df_sources_bars], ignore_index=True)
# # df_all_last    = filter_last_layer(df_all_combo)
# # print("Categories in bar_group:", sorted(df_all_last["bar_group"].dropna().unique().tolist()))

# # # Make sure tuned doesn’t have a participant_source_id (or it will be relabeled as source-*)
# # print("behaviortuned has participant_source_id column?",
# #       "participant_source_id" in df_behaviortuned.columns)

# # # Show a quick pivot of mean correlation per category to see if values are actually identical upstream
# # print(
# #     df_all_last.groupby(["model","participant_id","bar_group"])["correlation"]
# #       .mean().reset_index()
# #       .sort_values(["model","participant_id","bar_group"])
# #       .head(40)
# # )

# # # Also check duplicates: if tuned and each source row are literally identical, you’ll see it here
# # dups = (df_all_last
# #         .groupby(["model","participant_id","target","layer","bar_group"])["correlation"]
# #         .agg(["count","mean"])
# #         .reset_index()
# #        )
# # print("Any exact duplicates per (model, pid, target, layer, bar_group)?",
# #       (dups["count"]>1).any())

# # for unfreeze_last_n in df_behaviortuned_clean["unfreeze_last_n"].unique():
# #     for behavior_embeddings in df_behaviortuned_clean["behavior_embeddings"].unique():
# #         for n_components in df_behavior_clean["n_components"].unique():
# #             # print(f"behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
# #             print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
# #             df_behaviortuned_filtered = df_filter(
# #                 df_behaviortuned_clean,
# #                 filters={
# #                     "ds": "sagar2023",
# #                     # "unfreeze_last_n": unfreeze_last_n,
# #                     "behavior_embeddings": behavior_embeddings,
# #                     "n_components": n_components,
# #                 })
# #             df_behavior_filtered = df_filter(
# #                 df_behavior_clean,
# #                 filters={"ds": "sagar2023"}
# #             )

# #             plot_correlation_bars_by_participant_grid(
# #                 df_all_last,
# #                 corr_col="correlation",
# #                 target_col="target",
# #                 pid_col="participant_id",
# #                 model_col="model",
# #                 type_col="type",
# #                 bar_group_col="bar_group",   # <<< NEW: shows frozen, fine-tuned, and source-* bars
# #                 pval_col="p_value_correlation",
# #                 pval_thresh=p_val,
# #                 sort_targets="mean",
# #                 save_dir=out_dir,
# #                 filename_stub="fmri_last_with_sources",
# #                 show=False,
# #             )
# #             plot_avg_over_descriptors_by_participant_grid(
# #                 df_all_last,
# #                 value_col="correlation",
# #                 target_col="target",
# #                 pid_col="participant_id",
# #                 model_col="model",
# #                 type_col="type",
# #                 bar_group_col="bar_group",   # <<< NEW: shows frozen, fine-tuned
# #                 pval_col="p_value_correlation",
# #                 pval_thresh=p_val,      # optional
# #                 agg="mean",            # or "median"
# #                 error="sem",           # "std" or None
# #                 min_targets=3,
# #                 save_dir=out_dir,
# #                 filename_stub="fmri_avg_over_desc_last_with_sources",
# #                 show=False
# #             )



# #             #fMRI