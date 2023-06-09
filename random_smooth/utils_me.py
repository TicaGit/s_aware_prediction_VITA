from contextlib import contextmanager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from trajnetbaselines.lstm.utils import canvas, get_hex, seperate_xy, is_stationary, good_list




def draw_three_tensor(filename, noisy_pred, pred,  gt, collision_point_neighbor=None, collision_point_main=None):
    """
    order : noisy pred, untouched pred, ground truth
    """
    gt = gt.permute(1, 0, 2).tolist()
    pred = pred.permute(1, 0, 2).tolist()
    noisy_pred = noisy_pred.permute(1, 0, 2).tolist()
    with paths3(noisy_pred, pred, gt, filename, collision_point_neighbor, collision_point_main):
        pass

def draw_with_bounds(filename, m_pred, lb, ub, r_pred = None, collision_point_neighbor=None, collision_point_main=None):
    m_pred = m_pred.permute(1, 0, 2).tolist()
    if r_pred is not None:
        r_pred = r_pred.permute(1, 0, 2).tolist()
    lb = lb.permute(1, 0, 2).tolist()
    ub = ub.permute(1, 0, 2).tolist()
    with paths_bounds(m_pred, lb, ub, r_pred, filename, collision_point_neighbor, collision_point_main):
        pass



@contextmanager
def paths3(noisy_pred, pred, ground_t, output_file=None, collision_point_neighbor=None, collision_point_main=None):
    """Context to plot paths."""

    l1, l2 = 8, 8
    with canvas(output_file, figsize=(l1, l2)) as ax:
        #ax.grid(linestyle='dotted')
        #ax.set_aspect(1.0 , 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        start_symbol = 'o'
        end_symbol = 's'

        yield ax

        #
        obs_len = 9
        # other tracks

        len_limit = 21
        only_primary = False
        m_size_p = 3
        m_size_other = 4

        #colors        
        noisy_pred_color = get_hex(220,20,60) #red
        pred_color = get_hex(240, 240, 30) #yellowish
        ground_t_color = get_hex(143,147,83) #green 
        other_color = get_hex(119, 119, 119) #grey
        
        
        ## NOISY prediction ##
        for cnt, agent_path in enumerate(noisy_pred):
            
            xs, ys = seperate_xy(agent_path)
            #pdb.set_trace()
            if cnt > 0 and is_stationary(xs, ys):
              continue
            if len(good_list(xs)) < len_limit:
                continue
            if only_primary and cnt > 0:
                continue
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color=noisy_pred_color , marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=noisy_pred_color , marker=end_symbol, linestyle='None')

                ax.plot(xs[:obs_len], ys[:obs_len], color=noisy_pred_color , linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=noisy_pred_color , linestyle='dotted')
                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color=noisy_pred_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color=noisy_pred_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)

            else:
                xs = good_list(xs)
                ys = good_list(ys)
                ax.plot(xs[0:1], ys[0:1], color=other_color, marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=other_color, marker=end_symbol, linestyle='None')

                
                ax.plot(xs[:obs_len], ys[:obs_len], color=other_color, linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=other_color, linestyle='dotted')

                for j in range(1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color= other_color, marker='o', linestyle='None', zorder=0.9,
                            markersize=m_size_other)

        ## untouched prediction ##
        for cnt, agent_path in enumerate(pred):
            
            xs, ys = seperate_xy(agent_path)
            #pdb.set_trace()
            if cnt > 0 and is_stationary(xs, ys):
              continue
            if len(good_list(xs)) < len_limit:
                continue
            if only_primary and cnt > 0:
                continue
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color=pred_color , marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=pred_color , marker=end_symbol, linestyle='None')

                ax.plot(xs[:obs_len], ys[:obs_len], color=pred_color , linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=pred_color , linestyle='dotted')
                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color=pred_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color=pred_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)

            else:
                continue
                # xs = good_list(xs)
                # ys = good_list(ys)
                # ax.plot(xs[0:1], ys[0:1], color=other_color, marker=start_symbol, linestyle='None')
                # ax.plot(xs[-1:], ys[-1:], color=other_color, marker=end_symbol, linestyle='None')

                
                # ax.plot(xs[:obs_len], ys[:obs_len], color=other_color, linestyle='-')
                # ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=other_color, linestyle='dotted')

                # for j in range(1, len(xs) - 1):
                #     ax.plot(xs[j], ys[j], color= other_color, marker='o', linestyle='None', zorder=0.9,
                #             markersize=m_size_other)


        ### groud_truth ###
        obs_len = 9
        # other tracks
        for cnt, agent_path in enumerate(ground_t):
            xs, ys = seperate_xy(agent_path)

            # markers
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color=ground_t_color, marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=ground_t_color, marker=end_symbol, linestyle='None')
                # track

                ax.plot(xs[:obs_len], ys[:obs_len], color=ground_t_color, linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=ground_t_color, linestyle='dotted')

                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color=ground_t_color, marker='o', linestyle='None', zorder=1.2, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color=ground_t_color, marker='o', linestyle='None', zorder=1.2, markersize=m_size_p)
            else:
                continue


        orange_color = get_hex(210,97,42)
        if collision_point_neighbor != None and collision_point_main != None:
            x1 = collision_point_neighbor[0]
            y1 = collision_point_neighbor[1]
            x2 = collision_point_main[0]
            y2 = collision_point_main[1]
            dis = (x1 - x2)**2 + (y1 - y2)**2
            if dis < 0.09:
              ax.plot(collision_point_neighbor[0], collision_point_neighbor[1], color=noisy_pred_color, marker='o', linestyle='None', zorder=0.9, markersize=m_size_p + 1.0)
              ax.plot(collision_point_main[0], collision_point_main[1], color=noisy_pred_color, marker='o', linestyle='None', zorder=0.9, markersize=m_size_p + 1.0)


        # frame
        ax.set_facecolor(color = get_hex(233,242,245) )


@contextmanager
def paths_bounds(m_pred, lb, ub, r_pred, output_file = None, collision_point_neighbor=None, collision_point_main=None):
    """Context to plot paths."""

    l1, l2 = 8, 8
    with canvas(output_file, figsize=(l1, l2)) as ax:
        #ax.grid(linestyle='dotted')
        #ax.set_aspect(1.0 , 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        #new
        ax.set_aspect('equal', adjustable='box')
        #plt.axis('scaled')
                      
        start_symbol = 'o'
        end_symbol = 's'

        yield ax

        #
        obs_len = 9
        # other tracks

        len_limit = 21
        only_primary = False
        m_size_p = 3
        m_size_other = 4

        #colors        
        mean_pred_color = get_hex(220,20,60) #red
        other_color = get_hex(220, 120, 80) #redish
        real_color = get_hex(143,147,83) #green 

        box_color_border = get_hex(119, 119, 119) #grey
        box_color = get_hex(90, 90, 90) #light grey
        
        
        ## MEAN prediction ##
        for cnt, (agent_path, agent_lb, agent_ub) in enumerate(zip(m_pred, lb, ub)):

            #breakpoint()
            
            xs, ys = seperate_xy(agent_path)
            #pdb.set_trace()
            if cnt > 0 and is_stationary(xs, ys):
              continue  
            #HERE : Only draw agent with len = 21 = 9 + 12
            if len(good_list(xs)) < len_limit:
                continue
                pass
            if only_primary and cnt > 0:
                continue
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color=mean_pred_color , marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=mean_pred_color , marker=end_symbol, linestyle='None')

                ax.plot(xs[:obs_len], ys[:obs_len], color=mean_pred_color , linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=mean_pred_color , linestyle='dotted')
                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color=mean_pred_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color=mean_pred_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)

            else:
                xs = good_list(xs)
                ys = good_list(ys)
                ax.plot(xs[0:1], ys[0:1], color=other_color, marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=other_color, marker=end_symbol, linestyle='None')

                
                ax.plot(xs[:obs_len], ys[:obs_len], color=other_color, linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=other_color, linestyle='dotted')

                for j in range(1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color= other_color, marker='o', linestyle='None', zorder=0.9,
                            markersize=m_size_other)

            ## boxes ##
            xlb, ylb = seperate_xy(agent_lb)
            xub, yub = seperate_xy(agent_ub)

            draw_all = True
            if cnt == 0 or draw_all: #
                for xl, yl, xu, yu in zip(xlb, ylb, xub, yub):
                    width = xu - xl
                    height = yu - yl
                    if width<=0 or height<=0:
                        raise ValueError("bounds are wrong")
                    ax.add_patch(Rectangle((xl, yl), width, height, 
                                           facecolor = box_color, alpha = 0.2,
                                           edgecolor = box_color_border))
        #breakpoint()
        if r_pred is not None:
            for cnt, agent_path in enumerate(r_pred):
                xs, ys = seperate_xy(agent_path)
                #pdb.set_trace()
                if cnt > 0 and is_stationary(xs, ys):
                    continue  
                #HERE : Only draw agent with len = 21 = 9 + 12
                if len(good_list(xs)) < len_limit:
                    continue
                if only_primary and cnt > 0:
                    continue
                ax.plot(xs[0:1], ys[0:1], color=real_color , marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=real_color , marker=end_symbol, linestyle='None')

                ax.plot(xs[:obs_len], ys[:obs_len], color=real_color , linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=real_color , linestyle='dotted')
                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color=real_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color=real_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)


        orange_color = get_hex(210,97,42)
        if collision_point_neighbor != None and collision_point_main != None:
            x1 = collision_point_neighbor[0]
            y1 = collision_point_neighbor[1]
            x2 = collision_point_main[0]
            y2 = collision_point_main[1]
            dis = (x1 - x2)**2 + (y1 - y2)**2
            if dis < 0.09:
              ax.plot(collision_point_neighbor[0], collision_point_neighbor[1], color=mean_pred_color, marker='o', linestyle='None', zorder=0.9, markersize=m_size_p + 1.0)
              ax.plot(collision_point_main[0], collision_point_main[1], color=mean_pred_color, marker='o', linestyle='None', zorder=0.9, markersize=m_size_p + 1.0)


        # frame
        ax.set_facecolor(color = get_hex(233,242,245) )
