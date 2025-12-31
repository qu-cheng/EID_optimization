library(readxl)
library(dplyr)
library(ggplot2)
library(patchwork)
library(ggsci)
library(cowplot)

file_path <- "C:/Users/wangx/Desktop/manuscript/strategy_comprison_RF.xlsx"
sheets <- c("B. Modular", "C. Scale-free", "D. University", "E. High school", "F. Facebook", "G. Wild bird")
#sheets <- c("A. Modular", "B. Scale-free", "C. University", "D. High school", "E. Facebook", "F. Wild bird")

list_df <- lapply(sheets, function(sh) {
  df <- read_excel(file_path, sheet = sh) %>%
    mutate(sheet = sh,
           `Number of sentinel nodes` = as.numeric(`Number of sentinel nodes`),
           mean = as.numeric(mean),
           std = as.numeric(std))
  df
})
all_df <- bind_rows(list_df)

all_df <- all_df %>% filter(strategy != "RFSM-6%")
all_df <- all_df %>% filter(strategy != "Random")

non_complete <- all_df %>% filter(strategy != "Complete")
global_x_min <- min(non_complete$`Number of sentinel nodes`, na.rm = TRUE)
global_x_max <- 8.5

strategy_levels <- unique(all_df$strategy)
if(!"Complete" %in% strategy_levels) strategy_levels <- c(strategy_levels, "Complete")

pal_colors <- ggsci::pal_nejm("default")(length(strategy_levels))
names(pal_colors) <- strategy_levels

plot_sheet <- function(df_all, sheet_name, x_offset = 0.2) {
  df <- df_all %>% filter(sheet == sheet_name)
  complete_row <- df %>% filter(strategy == "Complete") %>% slice(1)
  others <- df %>% filter(strategy != "Complete")
  
  # 给每个策略添加固定的水平偏移量
  strategy_order <- c("Complete", "GA", "Greedy","RFSM", "Modular", "Global")
  strategy_offsets <- setNames(seq(-x_offset, x_offset, length.out = length(strategy_order)), strategy_order)
  
  # Complete 基准线
  complete_base <- NULL
  if (nrow(complete_row) == 1) {
    complete_base <- data.frame(
      x = c(global_x_min, global_x_max),
      mean = rep(complete_row$mean[1], 2),
      std  = rep(complete_row$std[1], 2),
      strategy = "Complete"
    )
  }
  
  p <- ggplot() +
    # Complete 灰带 + 虚线
    { if(!is.null(complete_base))
      geom_ribbon(data = complete_base,
                  aes(x = x, ymin = mean - std, ymax = mean + std),
                  inherit.aes = FALSE,
                  fill = pal_colors["Complete"], alpha = 0.18,
                  show.legend = FALSE)
    } +
    { if(!is.null(complete_base))
      geom_line(data = complete_base,
                aes(x = x, y = mean, color = "Complete", linetype = "Complete"),
                size = 0.9, show.legend = TRUE)
    } +
    geom_point(data = others,
               aes(x = `Number of sentinel nodes` + strategy_offsets[strategy], 
                   y = mean, color = strategy, shape = strategy),
               size = 2) +
    geom_errorbar(data = others,
                  aes(x = `Number of sentinel nodes` + strategy_offsets[strategy], 
                      ymin = mean - std, ymax = mean + std, color = strategy),
                  width = 0.1, size = 0.5) +
    scale_color_manual(values = pal_colors[strategy_order], breaks = strategy_order) +  # 控制颜色顺序
    scale_linetype_manual(values = setNames(c("dashed", rep("solid", length(strategy_order) - 1)), strategy_order), 
                          guide = "none") + 
    scale_shape_manual(values = setNames(c(NA, rep(16, length(strategy_order) - 1)), strategy_order)) +  # 控制形状
    coord_cartesian(xlim = c(global_x_min, global_x_max)) +
    labs(title = sheet_name, x = NULL, y = NULL) +
    theme_bw(base_size = 14) +
    theme(
      panel.border = element_blank(),
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", size = 0.3),
      legend.title = element_blank(),
      #legend.position = 'none',
      plot.title = element_text(hjust = 0, vjust = 1, size = 14, face = "bold"),
      axis.line.x = element_line(color = "black"),
      axis.line.y = element_line(color = "black")
    ) +
    guides(shape = "none")
  
  return(p)
}

# Generate plots with fixed horizontal offsets (no jitter, but slight separation)
plots <- lapply(sheets, function(sh) plot_sheet(all_df, sh))

# Apply changes to the plots (no jitter or line, adjust for first three)
plots[1:3] <- lapply(plots[1:3], function(p) {
  p + theme(axis.text.x = element_blank())
})

# Combine the plots using cowplot
combined <- (plots[[1]] + plots[[2]] + plots[[3]]) /
  (plots[[4]] + plots[[5]] + plots[[6]]) + 
  theme(legend.position = "none") +
  plot_layout(guides = "collect")
combined <- ggdraw() +
  draw_plot(combined, x = 0.03, y = 0.03, width = 0.94, height = 0.94) +
  draw_label("Number of sentinel nodes", x = 0.45, y = 0.02, vjust = 0, angle = 0, size = 13) +
  draw_label("Surveillance performance", x = 0.02, y = 0.5, vjust = 1, angle = 90, size = 13) +
  theme(plot.background = element_rect(fill = "white", colour = NA))
combined

#ggsave("D:/figure/strategy_performance_random.png", combined, width = 12, height = 8, dpi = 320)

# 提取图例的函数
get_legend <- function(p) {
  tmp <- ggplot_gtable(ggplot_build(p))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  tmp$grobs[[leg]]
}
legend_plot <- get_legend(plots[[1]])
plot_legend <- ggdraw() +
  draw_plot(legend_plot, x = 0, y = 0, width = 1, height = 1) +
  theme(plot.background = element_rect(fill = "white", colour = NA))
plot_legend

plot_sheet <- function(df_all, sheet_name, x_offset = 0.25) {
  df <- df_all %>% filter(sheet == sheet_name)
  complete_row <- df %>% filter(strategy == "Complete") %>% slice(1)
  others <- df %>% filter(strategy != "Complete")
  
  # 给每个策略添加固定的水平偏移量
  strategy_order <- c("Complete", "GA", "Greedy","RFSM", "Modular", "Global")
  strategy_offsets <- setNames(seq(-x_offset, x_offset, length.out = length(strategy_order)), strategy_order)
  
  # Complete 基准线
  complete_base <- NULL
  if (nrow(complete_row) == 1) {
    complete_base <- data.frame(
      x = c(global_x_min, global_x_max),
      mean = rep(complete_row$mean[1], 2),
      std  = rep(complete_row$std[1], 2),
      strategy = "Complete"
    )
  }
  
  p <- ggplot() +
    # Complete 灰带 + 虚线
    { if(!is.null(complete_base))
      geom_ribbon(data = complete_base,
                  aes(x = x, ymin = mean - std, ymax = mean + std),
                  inherit.aes = FALSE,
                  fill = pal_colors["Complete"], alpha = 0.18,
                  show.legend = FALSE)
    } +
    { if(!is.null(complete_base))
      geom_line(data = complete_base,
                aes(x = x, y = mean, color = "Complete", linetype = "Complete"),
                size = 0.9, show.legend = TRUE)
    } +
    geom_point(data = others,
               aes(x = `Number of sentinel nodes` + strategy_offsets[strategy], 
                   y = mean, color = strategy, shape = strategy),
               size = 2) +
    geom_errorbar(data = others,
                  aes(x = `Number of sentinel nodes` + strategy_offsets[strategy], 
                      ymin = mean - std, ymax = mean + std, color = strategy),
                  width = 0.1, size = 0.5) +
    scale_color_manual(values = pal_colors[strategy_order], breaks = strategy_order) +  # 控制颜色顺序
    scale_linetype_manual(values = setNames(c("dashed", rep("solid", length(strategy_order) - 1)), strategy_order), 
                          guide = "none") + 
    scale_shape_manual(values = setNames(c(NA, rep(16, length(strategy_order) - 1)), strategy_order)) +  # 控制形状
    coord_cartesian(xlim = c(global_x_min, global_x_max)) +
    labs(title = sheet_name, x = NULL, y = NULL) +
    theme_bw(base_size = 13) +
    theme(
      panel.border = element_blank(),
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", size = 0.3),
      legend.title = element_blank(),
      legend.position = 'none',
      plot.title = element_text(hjust = 0, vjust = 1, size = 14, face = "bold"),
      axis.line.x = element_line(color = "black"),
      axis.line.y = element_line(color = "black")
    ) +
    guides(shape = "none")
  
  return(p)
}

# Generate plots with fixed horizontal offsets (no jitter, but slight separation)
plots <- lapply(sheets, function(sh) plot_sheet(all_df, sh))

# Apply changes to the plots (no jitter or line, adjust for first three)
plots[1:3] <- lapply(plots[1:3], function(p) {
  p + theme(axis.text.x = element_blank())
})

# Combine the plots using cowplot
combined1 <- (plots[[1]] + plots[[2]] + plots[[3]]) /
  (plots[[4]] + plots[[5]] + plots[[6]]) + 
  theme(legend.position = "none") +
  plot_layout(guides = "collect")
combined1 <- ggdraw() +
  draw_plot(combined1, x = 0.03, y = 0.03, width = 0.94, height = 0.94) +
  draw_label("Number of sentinel nodes", x = 0.45, y = 0.02, vjust = 0, angle = 0, size = 13) +
  draw_label("Surveillance performance", x = 0.02, y = 0.5, vjust = 1, angle = 90, size = 13) +
  theme(plot.background = element_rect(fill = "white", colour = NA))
combined1


library(scales)
file_path <- "C:/Users/wangx/Desktop/manuscript/manuscript-list/0728/comparsion.xlsx"
df <- read_excel(file_path, sheet = "Sheet5")

# 自定义颜色（与示例图一致）
custom_colors <- c(
  "RFSM" = "#BC3C29FF",
  "Greedy" = "#0072B5FF",
  #"GA" = "#E18727FF",
  "Global" = "#20854EFF",
  "Random" = "#FFDC91FF",
  "Modular" = "#7876B1FF"
)

p2 <- ggplot(df,
             aes(x = `Network size`,
                 y = time,
                 color = strategy,
                 group = strategy)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = time - std, ymax = time + std),
                width = 10, alpha = 0.8) +
  scale_color_manual(values = custom_colors) +
  labs(
    x = "Network size",
    y = "Running time (s)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "white", colour = NA),
    panel.grid       = element_blank(),
    panel.border     = element_blank(),
    axis.line        = element_line(color = "black", size = 0.3),
    axis.ticks       = element_line(color = "black"),
    axis.title.x = element_text(size = 13, margin = margin(t = 7)),
    axis.title.y = element_text(size = 13),
    legend.title     = element_blank(),
    legend.position  = 'none',
    panel.background = element_blank(),
    axis.line.y.right= element_blank(),
    axis.line.x.top  = element_blank()
  ) 
  #annotate("text", x = -1, y = 25, label = "A. Comparison of running time", size = 5, hjust = 0, fontface = "bold")

print(p2)


p2_draw <- ggdraw() +
  theme(plot.background = element_rect(fill = "white", colour = NA)) +
  draw_plot(p2,x = 0.03, y = -0.10,width  = 1,height = 0.95)

final <- plot_grid((p2_draw/plot_legend), combined1,
                   ncol = 2, rel_widths = c(1.15, 3)) +
  annotate("text", x = 0.05, y = 0.935, label = "A. Comparison of running time", size = 5, hjust = 0, fontface = "bold")

final
