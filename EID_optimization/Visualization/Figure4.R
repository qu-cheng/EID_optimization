library(tidyverse)
library(patchwork)
library(cowplot)

df <- read_csv("D:/data_generating/importance_0.3/model_performance.csv")


ndcg_at_k <- function(act, prd, k){
  
  n <- length(act)
  if(k <= 0 || k > n) return(NA_real_)
  ord  <- order(prd, decreasing = TRUE)
  act  <- act[ord]
  dcg <- sum(act[1:k] / log2(seq_len(k) + 1))
  idcg <- sum(sort(act, decreasing = TRUE)[1:k] / log2(seq_len(k) + 1))
  if(idcg == 0) 0 else dcg / idcg
}

target_ids <- c(1000, 1001, 1002, 1003, 1004)

ndcg_values <- map_dbl(target_ids, function(id){
  sub <- df %>% filter(network_id == id)
  k   <- round(0.3 * nrow(sub)) 
  ndcg_at_k(sub$actual, sub$pred, k)
})

setNames(ndcg_values, target_ids)
ndcg_values <- c(0.930, 0.912, 0.964, 0.913, 0.967)
meta_df <- tibble(
  id    = target_ids,
  title = titles,
  ndcg  = ndcg_values
)

xlim <- c(0, 1)
ylim <- c(0, 0.35)

my_theme <- function()
  theme_bw(base_size = 14) +
  theme(
    panel.grid        = element_blank(),
    panel.border      = element_blank(),
    axis.line         = element_line(colour = "black"),
    legend.position   = "none",
    plot.title        = element_text(hjust = 0, vjust = 1, size = 12, face = "bold"),
    plot.margin       = margin(t = 6, r = 8, b = 6, l = 8)
  )

p1 <- ggplot(df, aes(x = actual, y = pred)) +
  geom_point(aes(colour = dataset, alpha = dataset), size = 1.5) +
  geom_vline(xintercept = 0.3, linetype = "dashed", colour = "red", size = 0.5) +  # 添加这行
  scale_x_continuous(limits = xlim, breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = ylim, breaks = seq(0, 1, 0.1)) +
  scale_alpha_manual(values = c("train" = 0.7, "test" = 0.5)) +
  my_theme() +
  labs(x = NULL, y = NULL, title = "A. Model performance") +
  theme(legend.position = c(1.01, -0.055),
        legend.justification = c(1, 0),
        legend.background = element_rect(fill = NA, colour = NA),
        legend.title = element_blank())

titles   <- c("B. Scale-free", "C. University", "D. High school",
              "E. Facebook", "F. Wildbird")

col_func <- function(dat)
  ifelse(dat$actual <= 0.3 & dat$pred <= 0.3,
         "grey30", 
         "grey30")

p_rest <- pmap(list(meta_df$id, meta_df$title, meta_df$ndcg),
               function(id, ttl, ndcg_val){
                 df_sub <- df %>% filter(network_id == id)
                 ggplot(df_sub, aes(actual, pred)) +
                   geom_point(shape = 1, size = 2.5,
                              colour = col_func(df_sub)) +
                   #geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "red") +
                   scale_x_continuous(limits = xlim, breaks = seq(0, 1, 0.2)) +
                   scale_y_continuous(limits = ylim, breaks = seq(0, 1, 0.1)) +
                   my_theme() +
                   labs(x = NULL, y = NULL, title = ttl) +
                   annotate("text",
                            x = -Inf, y = Inf,
                            label = sprintf("NDCG@30%%: %.3f", ndcg_val),
                            hjust = -0.2, vjust = 1,
                            size = 3.5, fontface = "bold")
               })

p1_top    <- p1 + theme(axis.text.x = element_blank())
p2_top    <- p_rest[[1]] + theme(axis.text.x = element_blank(),
                                 axis.text.y = element_blank())
p3_top    <- p_rest[[2]] + theme(axis.text.x = element_blank(),
                                 axis.text.y = element_blank())

p4_bottom <- p_rest[[3]]
p5_bottom <- p_rest[[4]] + theme(axis.text.y = element_blank())
p6_bottom <- p_rest[[5]] + theme(axis.text.y = element_blank())

p_grid <- wrap_plots(
  p1_top, p2_top, p3_top,
  p4_bottom, p5_bottom, p6_bottom,
  ncol = 3
)

final <- ggdraw() +
  draw_plot(p_grid, x = 0.03, y = 0.03, width = 0.94, height = 0.94) +
  theme(plot.background = element_rect(fill = "white", colour = NA)) +
  draw_label("Observed ranking",
             x = 0.5, y = 0.01,
             vjust = 0, size = 14) +
  draw_label("Predicted ranking",
             x = 0.01, y = 0.5,
             angle = 90, vjust = 1, size = 14)

print(final)

