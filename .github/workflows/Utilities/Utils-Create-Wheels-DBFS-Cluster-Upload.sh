# The Script Will Ingest Parameters File In Order To Determine Location Of Setup.py Files.
# Each Setup.py Relates To The Creation Of A New Wheel File, Which Will Be Saved In 
# DBFS In A Folder Corresponding To The Cluster The Wheel File Is To Be Uploaded To. 

echo "Import Wheel Dependencies"
python -m pip install --upgrade pip
python -m pip install flake8 pytest pyspark pytest-cov requests
pip3 install -r ./src/pipelines/dbkframework/requirements.txt
python -m pip install --user --upgrade setuptools wheel
sudo apt-get install pandoc


echo "Ingest JSON Environment File"
JSON=$( jq '.' .github/workflows/Pipeline_Param/$environment.json)
echo "${JSON}" | jq


for row in $(echo "${JSON}" | jq -r '.WheelFiles[] | @base64'); do
    _jq() {
        echo ${row} | base64 --decode | jq -r ${1}
    }

    CLUSTER_NAME=$(_jq '.wheel_cluster')
    setup_py_file_path=$(_jq '.setup_py_file_path')
    # We Are Removing Setup.py From The FilePath 'setup_py_file_path'.
    root_dir_file_path=${setup_py_file_path%/*}
    
    echo "Wheel File Destined For Cluster: $CLUSTER_NAME "
    echo "Location Of Setup.py File For Wheel File Creation; $setup_py_file_path"
    
    cd $root_dir_file_path

    # Create The Wheel File
    python setup.py sdist bdist_wheel
    
    cd dist 
    ls
    wheel_file_name=$( ls -d -- *.whl )
    echo "Wheel File Name: $wheel_file_name"

    # Install Wheel File
    ls
    echo "$root_dir_file_path/dist/$wheel_file_name"
    echo "Wheel File Name (should end in .whl): $wheel_file_name"

    az synapse workspace-package upload \
        --file "$wheel_file_name" \
        --workspace-name "synapsewsdeploychd"


    #$package = New-AzSynapseWorkspacePackage -WorkspaceName workspace1 -Package "C:\xxx-2.0.2-py3.whl"
    # Update-AzSynapseSparkPool -WorkspaceName workspace1 -Name pool2 -PackageAction Add -Package $package
    

done



