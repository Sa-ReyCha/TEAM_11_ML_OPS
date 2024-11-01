```mermaid
flowchart TD
	node1["data_load"]
	node2["data_separation"]
	node3["scale_data"]
	node4["split_data"]
	node5["train"]
	node3-->node5
	node4-->node3
	node6["file.txt.dvc"]
```