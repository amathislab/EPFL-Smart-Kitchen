#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License.
# A copy is included in LICENSE.
#
# Incorporates code adapted from MoMask by EricGuo5513
# Original work Copyright (c) 2023 by EricGuo5513
# Source: https://github.com/EricGuo5513/momask-codes
# Originally licensed under MIT License
#
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import *
from utils.motion_process import recover_from_ric


@torch.no_grad()
def evaluation_vqvae(
    out_dir,
    val_loader,
    net,
    writer,
    ep,
    best_fid,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    save=True,
    draw=True,
):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = (
            batch
        )

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, motion, m_length
        )
        bs, seq = motion.shape[0], motion.shape[1]

        pred_pose_eval = net(motion)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"
        % (
            ep,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
        )
    )
    print(msg)

    if draw:
        writer.add_scalar("./Test/FID", fid, ep)
        writer.add_scalar("./Test/Diversity", diversity, ep)
        writer.add_scalar("./Test/top1", R_precision[0], ep)
        writer.add_scalar("./Test/top2", R_precision[1], ep)
        writer.add_scalar("./Test/top3", R_precision[2], ep)
        writer.add_scalar("./Test/matching_score", matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw:
            print(msg)
        best_fid = fid
        if save:
            torch.save(
                {"vq_model": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_fid.tar"),
            )

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!" % (
            best_div,
            diversity,
        )
        if draw:
            print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (
            best_top1,
            R_precision[0],
        )
        if draw:
            print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (
            best_top2,
            R_precision[1],
        )
        if draw:
            print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (
            best_top3,
            R_precision[2],
        )
        if draw:
            print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (
            best_matching,
            matching_score_pred,
        )
        if draw:
            print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save(
                {"vq_model": net.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_mm.tar"),
            )


    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = (
            batch
        )

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, motion, m_length
        )
        bs, seq = motion.shape[0], motion.shape[1]


        pred_pose_eval = net(motion)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(
                torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint
            )
            pred = recover_from_ric(
                torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint
            )

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]


        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f"
        % (
            repeat_id,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            mpjpe,
        )
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, mpjpe


@torch.no_grad()
def evaluation_vqvae_plus_l1(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = (
            batch
        )

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, motion, m_length
        )
        bs, seq = motion.shape[0], motion.shape[1]


        pred_pose_eval, loss_commit, perplexity = net(motion)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(
                torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint
            )
            pred = recover_from_ric(
                torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint
            )
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"
        % (
            repeat_id,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            l1_dist,
        )
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_res_plus_l1(
    val_loader, vq_model, res_model, repeat_id, eval_wrapper, num_joint, do_vq_res=True
):
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = (
            batch
        )

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, motion, m_length
        )
        bs, seq = motion.shape[0], motion.shape[1]


        if do_vq_res:
            code_ids, all_codes = vq_model.encode(motion)
            if len(code_ids.shape) == 3:
                pred_vq_codes = res_model(code_ids[..., 0])
            else:
                pred_vq_codes = res_model(code_ids)
            pred_pose_eval = vq_model.decoder(pred_vq_codes)
        else:
            rec_motions, _, _ = vq_model(motion)
            pred_pose_eval = res_model(
                rec_motions
            ) 
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(
                torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint
            )
            pred = recover_from_ric(
                torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint
            )
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"
        % (
            repeat_id,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            l1_dist,
        )
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mardm(
    out_dir,
    val_loader,
    trans,
    vq_model,
    writer,
    ep,
    best_fid,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    plot_func,
    save_ckpt=False,
    save_anim=False,
):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [
            e for e in t2m_trans_state_dict.keys() if e.startswith("clip_model.")
        ]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            "t2m_transformer": t2m_trans_state_dict,
            "ep": ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2.5
    else:
        cond_scale = 4.5


    nb_sample = 0
    for batch in val_loader:
        (
            word_embeddings,
            pos_one_hots,
            clip_text,
            sent_len,
            pose,
            m_length,
            token,
            image,
        ) = batch
        m_length = m_length.cuda()
        image = image.cuda()

        bs, seq = pose.shape[:2]

        # (b, seqlen)
        pred_latents = trans.generate(
            clip_text, image, m_length // 4, time_steps, cond_scale, temperature=1
        )

        pred_motions = vq_model.decode(pred_latents)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
        )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar("./Test/FID", fid, ep)
    writer.add_scalar("./Test/Diversity", diversity, ep)
    writer.add_scalar("./Test/top1", R_precision[0], ep)
    writer.add_scalar("./Test/top2", R_precision[1], ep)
    writer.add_scalar("./Test/top3", R_precision[2], ep)
    writer.add_scalar("./Test/matching_score", matching_score_pred, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, "model", "net_best_fid.tar"), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = (
            f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        )
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = (
            f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        )
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = (
            f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        )
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = (
            f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        )
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, "animation", "E%04d" % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_mask_transformer(
    out_dir,
    val_loader,
    trans,
    vq_model,
    writer,
    ep,
    best_fid,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    plot_func,
    save_ckpt=False,
    save_anim=False,
):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [
            e for e in t2m_trans_state_dict.keys() if e.startswith("clip_model.")
        ]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            "t2m_transformer": t2m_trans_state_dict,
            "ep": ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    nb_sample = 0
    for batch in val_loader:
        (
            word_embeddings,
            pos_one_hots,
            clip_text,
            sent_len,
            pose,
            m_length,
            token,
            image,
        ) = batch
        m_length = m_length.cuda()
        image = image.cuda()

        bs, seq = pose.shape[:2]

        mids = trans.generate(
            clip_text, image, m_length // 4, time_steps, cond_scale, temperature=1
        )

        mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
        )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar("./Test/FID", fid, ep)
    writer.add_scalar("./Test/Diversity", diversity, ep)
    writer.add_scalar("./Test/top1", R_precision[0], ep)
    writer.add_scalar("./Test/top2", R_precision[1], ep)
    writer.add_scalar("./Test/top3", R_precision[2], ep)
    writer.add_scalar("./Test/matching_score", matching_score_pred, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, "model", "net_best_fid.tar"), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = (
            f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        )
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = (
            f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        )
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = (
            f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        )
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = (
            f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        )
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, "animation", "E%04d" % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer(
    out_dir,
    val_loader,
    trans,
    vq_model,
    writer,
    ep,
    best_fid,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    plot_func,
    save_ckpt=False,
    save_anim=False,
    cond_scale=2,
    temperature=1,
):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [
            e for e in res_trans_state_dict.keys() if e.startswith("clip_model.")
        ]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            "res_transformer": res_trans_state_dict,
            "ep": ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0


    nb_sample = 0
    for batch in val_loader:
        (
            word_embeddings,
            pos_one_hots,
            clip_text,
            sent_len,
            pose,
            m_length,
            token,
            image,
        ) = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()
        image = image.cuda()

        bs, seq = pose.shape[:2]

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(
                code_indices[..., 0],
                clip_text,
                image,
                m_length // 4,
                temperature=temperature,
                cond_scale=cond_scale,
            )

        pred_motions = vq_model.forward_decoder(pred_ids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
        )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar("./Test/FID", fid, ep)
    writer.add_scalar("./Test/Diversity", diversity, ep)
    writer.add_scalar("./Test/top1", R_precision[0], ep)
    writer.add_scalar("./Test/top2", R_precision[1], ep)
    writer.add_scalar("./Test/top3", R_precision[2], ep)
    writer.add_scalar("./Test/matching_score", matching_score_pred, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, "model", "net_best_fid.tar"), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = (
            f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        )
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = (
            f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        )
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = (
            f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        )
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = (
            f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        )
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, "animation", "E%04d" % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer_plus_l1(
    val_loader,
    vq_model,
    trans,
    repeat_id,
    eval_wrapper,
    num_joint,
    cond_scale=2,
    temperature=1,
    topkr=0.9,
    cal_l1=True,
):

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = (
            batch
        )
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]

        code_indices, all_codes = vq_model.encode(pose)

        pred_ids = trans.generate(
            code_indices[..., 0],
            clip_text,
            m_length // 4,
            topk_filter_thres=topkr,
            temperature=temperature,
            cond_scale=cond_scale,
        )

        pred_motions = vq_model.forward_decoder(pred_ids)

        if cal_l1:
            bgt = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
            bpred = val_loader.dataset.inv_transform(
                pred_motions.detach().cpu().numpy()
            )
            for i in range(bs):
                gt = recover_from_ric(
                    torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint
                )
                pred = recover_from_ric(
                    torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint
                )
                num_pose = gt.shape[0]
                l1_dist += F.l1_loss(gt, pred) * num_pose
                num_poses += num_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
        )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"
        % (
            repeat_id,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            l1_dist,
        )
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mardm_test(
    val_loader,
    vq_model,
    trans,
    repeat_id,
    eval_wrapper,
    time_steps,
    cond_scale,
    temperature,
    cal_mm=True,
):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = (
            batch
        )
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                pred_latents = trans.generate(
                    clip_text,
                    m_length // 4,
                    time_steps,
                    cond_scale,
                    temperature=temperature,
                )

                pred_motions = vq_model.decode(pred_latents)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    pred_motions.clone(),
                    m_length,
                )
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(
                motion_multimodality_batch, dim=1
            )  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            pred_latents = trans.generate(
                clip_text,
                m_length // 4,
                time_steps,
                cond_scale,
                temperature=temperature,
            )

            pred_motions = vq_model.decode(pred_latents)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
            )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_transformer_test(
    val_loader,
    vq_model,
    trans,
    repeat_id,
    eval_wrapper,
    time_steps,
    cond_scale,
    temperature,
    topkr,
    gsample=True,
    force_mask=False,
    cal_mm=True,
):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        (
            word_embeddings,
            pos_one_hots,
            clip_text,
            sent_len,
            pose,
            m_length,
            token,
            image,
        ) = batch
        m_length = m_length.cuda()
        image = image.cuda()

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(
                    clip_text,
                    image,
                    m_length // 4,
                    time_steps,
                    cond_scale,
                    temperature=temperature,
                    topk_filter_thres=topkr,
                    gsample=gsample,
                    force_mask=force_mask,
                )

                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    pred_motions.clone(),
                    m_length,
                )
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(
                motion_multimodality_batch, dim=1
            )  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(
                clip_text,
                image,
                m_length // 4,
                time_steps,
                cond_scale,
                temperature=temperature,
                topk_filter_thres=topkr,
                force_mask=force_mask,
            )

            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
            )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_transformer_test_plus_res(
    val_loader,
    vq_model,
    res_model,
    trans,
    repeat_id,
    eval_wrapper,
    time_steps,
    cond_scale,
    temperature,
    topkr,
    gsample=True,
    force_mask=False,
    cal_mm=True,
    res_cond_scale=5,
):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        (
            word_embeddings,
            pos_one_hots,
            clip_text,
            sent_len,
            pose,
            m_length,
            token,
            image,
        ) = batch
        m_length = m_length.cuda()
        image = image.cuda()

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(
                    clip_text,
                    image,
                    m_length // 4,
                    time_steps,
                    cond_scale,
                    temperature=temperature,
                    topk_filter_thres=topkr,
                    gsample=gsample,
                    force_mask=force_mask,
                )

                pred_ids = res_model.generate(
                    mids,
                    clip_text,
                    image,
                    m_length // 4,
                    temperature=1,
                    cond_scale=res_cond_scale,
                )

                pred_motions = vq_model.forward_decoder(pred_ids)


                et_pred, em_pred = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    pred_motions.clone(),
                    m_length,
                )
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(
                motion_multimodality_batch, dim=1
            )  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(
                clip_text,
                image,
                m_length // 4,
                time_steps,
                cond_scale,
                temperature=temperature,
                topk_filter_thres=topkr,
                force_mask=force_mask,
            )

            pred_ids = res_model.generate(
                mids,
                clip_text,
                image,
                m_length // 4,
                temperature=1,
                cond_scale=res_cond_scale,
            )

            pred_motions = vq_model.forward_decoder(pred_ids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
            )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_visualization(
    val_loader,
    vq_model,
    res_model,
    trans,
    repeat_id,
    eval_wrapper,
    time_steps,
    cond_scale,
    temperature,
    topkr,
    gsample=True,
    force_mask=False,
    cal_mm=True,
    res_cond_scale=5,
    plot_t2m=None,
    save_dir=None,
):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in tqdm(enumerate(val_loader)):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = (
            batch
        )
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(
                    clip_text,
                    m_length // 4,
                    time_steps,
                    cond_scale,
                    temperature=temperature,
                    topk_filter_thres=topkr,
                    gsample=gsample,
                    force_mask=force_mask,
                )

                pred_ids = res_model.generate(
                    mids,
                    clip_text,
                    m_length // 4,
                    temperature=1,
                    cond_scale=res_cond_scale,
                )

                pred_motions = vq_model.forward_decoder(pred_ids)


                et_pred, em_pred = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    pred_motions.clone(),
                    m_length,
                )
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(
                motion_multimodality_batch, dim=1
            )  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(
                clip_text,
                m_length // 4,
                time_steps,
                cond_scale,
                temperature=temperature,
                topk_filter_thres=topkr,
                force_mask=force_mask,
            )

            pred_ids = res_model.generate(
                mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale
            )

            pred_motions = vq_model.forward_decoder(pred_ids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
            )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

        if plot_t2m is not None and i % 10 == 0:
            data = pred_motions.detach().cpu().numpy()
            captions = [clip_text[k] for k in range(bs)]
            lengths = m_length.cpu().numpy()
            batch_save_dir = os.path.join(
                "example_data_new/{}".format(save_dir), "Batch%04d" % i
            )
            os.makedirs(batch_save_dir, exist_ok=True)
            plot_t2m(data, batch_save_dir, captions, lengths)

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality
