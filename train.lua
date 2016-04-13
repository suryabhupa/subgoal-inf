require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'xlua'
require 'image'

noutputs = 50

nfeats = 3
width = 210
height = 320
ninputs = nfeats*width*height

nstates = {64, 64, 128}
dims = {49, 77}
filtsize = 5
poolsize = 2 

function create_model()
	print('==> creating model')
	local model = nn.Sequential()

	local convnet = nn.Sequential()
	convnet:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize, filtsize))
	convnet:add(nn.ReLU())
	convnet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	convnet:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize, filtsize))
	convnet:add(nn.ReLU())
	convnet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	convnet:add(nn.View(nstates[2]*dims[1]*dims[2]))

	convnet:add(nn.Dropout(0.5))
	convnet:add(nn.Linear(nstates[2]*dims[1]*dims[2], nstates[3]))
	convnet:add(nn.ReLU())
	convnet:add(nn.Linear(nstates[3], noutputs)) 

	local onehot = nn.Sequential()
	onehot:add(nn.Linear(9, 9)) 

	local parallel = nn.ParallelTable()
	parallel:add(onehot)
	parallel:add(convnet)

	model:add(parallel)
	model:add(nn.JoinTable(1))

	-- Add more layers here for deconvolution

	print(model)
	return model

end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function get_action_data(file)
	-- read in the acations in the file, line by line
	print("==> getting action data")
	data = {}

	-- open file, check for errors
	fh, err = io.open(file) 
	if err then print("Error in opening file: " .. file) return end 

	-- loop while there are lines in the file
	while true do
		-- read line, break if nil
		local line = fh:read()
		if line == nil then break end

		-- each line is a string like "1 0 0 0 0 0 0"
		-- need to split string by space, and then concat to table
		-- then convert table to tensor, and add to overall data table
		inp = {}
		for i in string.gmatch(line, "%d") do
			table.insert(inp, tonumber(i))
		end
		table.insert(data, torch.Tensor(inp))
	end
	return data
end

function get_frame_data(dir)
	-- get all *.png files in the directory and sort them
	print("==> getting frame data")
	files = {}
	for file in paths.files(dir) do
		if file:find("png$") then
			-- note that this only appends the *filenames* to the table, not the actual images
			table.insert(files, paths.concat(dir, file))
		end
	end 

	table.sort(files, function (a,b) return a < b end) 
	
	return files
end

function make_training_data(action_data_file, frame_data_dir)
	print("==> preparing dataset")
	a_data = get_action_data(action_data_file)
	f_data = get_frame_data(frame_data_dir)

	return {a_data, f_data}
end

function test(model)
	
end

function main()
	print("==> starting main")
	model = create_model()
	training_data = make_training_data("ALE/doc/examples/record/game_actions.txt", "ALE/doc/examples/record/")
	a_data = training_data[1]
	f_data = training_data[2]

	model:training() -- put into training mode (dropout turns on)
	-- criterion = nn.MSECriterion()

	for j = 1,100 do
		-- one less than the last entry, because we compare to next element
		for i = 1,10867 do
			-- input = {a_data[i], image.load(f_data[i])}
			-- output = {image.load(f_data[(i+1)])}

			-- criterion:forward(model:forward(input), output)
			-- model:zeroGradParameters()
			-- model:backward(input, criterion:backward(model.output, output))
			-- model:updateParameters(0.01) 
		end
	end

	-- torch.save("cps/" .. os.time .. ".dat", model)
	-- torch.save("cps/" .. "1" .. ".dat", model)
end

main()

--[[
if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
--]]

