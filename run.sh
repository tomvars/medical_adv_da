docker run -it \
--gpus=all \
-v /home/tom/DomainAdaptationJournal:/home/tom/DomainAdaptationJournal \
-v /data2:/data2/ \
--shm-size 8G \
-p 8701 -p 8702 -p 8703 \
domain_adaptation_journal:latest