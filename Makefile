.SILENT:

sweep $(sweep-config-path):
	chmod -R +x tasks/
	tasks/sweep.sh $(sweep-config-path)