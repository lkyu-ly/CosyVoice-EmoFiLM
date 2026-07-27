# Issue tracker: Local Markdown

本仓库的规格（PRD）与实施票据以 Markdown 文件保存在 `.scratch/` 中。

## Conventions

- 每个功能或工作流使用独立目录：`.scratch/<feature-slug>/`。
- 规格文件固定为 `.scratch/<feature-slug>/spec.md`。
- 实施票据逐个写入 `.scratch/<feature-slug>/issues/<NN>-<slug>.md`，从 `01` 起按依赖顺序编号；不得合并为单一票据文件。
- 每个票据在文件顶部附近使用 `Status:` 记录 triage 状态，具体词汇见 `triage-labels.md`。
- 评论与讨论历史追加到文件末尾的 `## Comments` 小节。

## Publishing

当技能要求“发布到 issue tracker”时，在 `.scratch/<feature-slug>/` 下创建对应文件；目录不存在时可以创建。

当技能要求读取相关规格或票据时，读取用户给出的路径、编号或当前工作流目录中的对应文件。

## Wayfinding operations

- 地图文件：`.scratch/<effort>/map.md`。
- 子票据：`.scratch/<effort>/issues/<NN>-<slug>.md`。
- 阻塞关系：在文件顶部附近使用 `Blocked by:`，列出依赖票据编号；全部依赖为 `resolved` 后，票据才进入 frontier。
- 领取票据：工作开始前将 `Status:` 设置为 `claimed`。
- 完成票据：追加 `## Answer`，将 `Status:` 设置为 `resolved`，并在地图的 Decisions-so-far 中增加摘要与链接。
