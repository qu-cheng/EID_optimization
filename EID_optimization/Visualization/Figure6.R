library(readxl)
library(ggplot2)
library(dplyr)
library(patchwork)

# 读取数据
file_path <- "C:/Users/wangx/Desktop/manuscript/manuscript-list/0728/comparsion.xlsx"
df1 <- read_excel(file_path, sheet = "Sheet2")
df2 <- read_excel(file_path, sheet = "Sheet3")
df3 <- read_excel(file_path, sheet = "Sheet4")

# 添加网络类型标签
df1$network_type <- "A. Modular networks"
df2$network_type <- "B. Scale-free networks"
df3$network_type <- "C. Empirical networks"

# 重命名分类列以统一
df1 <- df1 %>% rename(category = number)
df2 <- df2 %>% rename(category = number)
df3 <- df3 %>% rename(category = network)

my_colors <- c(
  "Complete" = "#6F99ADFF",
  "6-RFSM" = "#FF6F0099",
  "6%-RFSM" = "#BC3C29FF"
)

# 通用绘图函数
plot_fun <- function(df, title_name,x_name,show_y=FALSE, show_legend = FALSE) {
  ggplot(df, aes(x = as.factor(category), y = performance, color = method)) +
    geom_point(position = position_dodge(width = 0.5), size = 2) +
    geom_errorbar(aes(ymin = performance - std, ymax = performance + std),
                  width = 0.2,
                  position = position_dodge(width = 0.5)) +
    scale_color_manual(values = my_colors) +
    labs(x = x_name, y = if (show_y) "Surveillance performance" else NULL, title = title_name, color = "Method") +
    theme_bw(base_size = 13) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(colour = "black"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0),
      legend.position = if (show_legend) c(0.25, 0.85) else "none", 
      legend.direction = "vertical", 
      legend.text = element_text(size = 11),
      legend.title = element_blank()
    )
}

p1 <- plot_fun(df1, "A. Modular networks","Network size", show_y = TRUE, show_legend = TRUE)
p2 <- plot_fun(df2, "B. Scale-free networks","Network size")
p3 <- plot_fun(df3, "C. Empirical networks", "Network")

final_plot <- (p1 | p2 | p3)

final_plot
