## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
HTML profiling visualization for LC Agent networks.

This module creates interactive timeline visualizations of profiling data,
similar to professional profiling tools like pyinstrument or Tracy.
"""

from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Union
import json
import os
from pathlib import Path

if TYPE_CHECKING:
    from ..runnable_network import RunnableNetwork
    from .profiling_utils import ProfilingFrame, ProfilingData


def create_profiling_html(
    network: Union["RunnableNetwork", Dict[str, Any]], output_path: str = "profiling.html", debug: bool = False
) -> str:
    """
    Create an HTML file with interactive profiling visualization.
    
    Args:
        network: Network with profiling data (RunnableNetwork object or dict)
        output_path: Path to save the HTML file
        
    Returns:
        Path to the created HTML file
    """
    # Convert RunnableNetwork to dict if needed
    if hasattr(network, "model_dump"):
        # It's a RunnableNetwork object, convert to dict
        network_dict = network.model_dump()
        network_name = network.__class__.__name__
    else:
        # It's already a dict
        network_dict = network
        network_name = network_dict.get("__node_type__", "Network")

    # Check for profiling data
    profiling = network_dict.get("profiling")
    if not profiling or not profiling.get("frames"):
        raise ValueError("No profiling data available in network")
    
    # Collect all frames and assign tracks
    frames_data = []

    def count_frame_depth(frame: Dict[str, Any]) -> int:
        """Count maximum depth of frame and its children."""
        children = frame.get("children", [])
        if not children:
            return 1
        return 1 + max(count_frame_depth(child) for child in children)

    def count_network_tracks(network_obj: Dict[str, Any]) -> int:
        """Count how many tracks a network needs based on its profiling frames."""
        profiling = network_obj.get("profiling", {})
        frames = profiling.get("frames", [])
        if not frames:
            return 0

        max_depth = 0
        for frame in frames:
            max_depth = max(max_depth, count_frame_depth(frame))
        return max_depth

    def collect_frames(frame: Dict[str, Any], track_offset: int = 0) -> None:
        """Collect frame data and assign to specific track."""
        start_time = frame.get("start_time")
        if start_time is None:
            return

        # If frame has no end time, skip it (incomplete frame)
        end_time = frame.get("end_time")
        if end_time is None:
            # Still process children in case they are complete
            children = frame.get("children", [])
            for i, child in enumerate(children):
                collect_frames(child, track_offset + 1)
            return

        # Create frame data
        frame_data = {
            "name": frame.get("name", ""),
            "type": frame.get("frame_type", ""),
            "start": start_time,
            "end": end_time,
            "duration": frame.get("duration"),
            "track": track_offset,  # Absolute track number
            "metadata": frame.get("metadata", {}).copy(),
        }

        # Add human-readable duration
        duration = frame.get("duration")
        if duration:
            if duration < 0.001:
                frame_data["duration_str"] = f"{duration * 1000000:.1f}μs"
            elif duration < 1:
                frame_data["duration_str"] = f"{duration * 1000:.1f}ms"
            else:
                frame_data["duration_str"] = f"{duration:.3f}s"
        else:
            frame_data["duration_str"] = "N/A"

        frames_data.append(frame_data)

        # Process children frames on next track
        children = frame.get("children", [])
        for child in children:
            collect_frames(child, track_offset + 1)

    # Debug counters
    networks_found = [0]
    nodes_checked = [0]

    def collect_network_frames(
        network_obj: Dict[str, Any], track_start: int = 0, network_name: str = None, level: int = 0
    ) -> int:
        """
        Recursively collect frames from a network and all its sub-networks.
        Returns the next available track number.
        """
        if not network_obj:
            return track_start

        networks_found[0] += 1
        if debug:
            print(f"{'  ' * level}Processing network: {network_name} (track_start={track_start})")

        current_track = track_start

        # Process this network's profiling frames
        profiling = network_obj.get("profiling", {})
        frames = profiling.get("frames", [])
        if frames:
            if debug:
                print(f"{'  ' * level}  Found {len(frames)} profiling frames")

            # Count tracks needed for this network
            tracks_needed = count_network_tracks(network_obj)
            if debug:
                print(f"{'  ' * level}  Needs {tracks_needed} tracks")

            if track_start > 0 and network_name:  # Add separator for nested networks
                frames_data.append(
                    {
                        "name": f"=== {network_name} ===",
                        "type": "separator",
                        "track": current_track,
                        "is_separator": True,
                    }
                )

            # Collect all frames starting at current track
            for frame in frames:
                collect_frames(frame, current_track)

            # Move to next available track
            current_track += tracks_needed

        elif debug:
            print(f"{'  ' * level}  No profiling data found")

        # Now recursively process all nodes to find sub-networks
        nodes = network_obj.get("nodes", [])
        if nodes:
            if debug:
                print(f"{'  ' * level}  Checking {len(nodes)} nodes for sub-networks")
            for i, node in enumerate(nodes):
                nodes_checked[0] += 1
                node_name = node.get("name") or node.get("__node_type__", "Unknown")

                # Check multiple ways a node can contain a network
                # 1. Node with subnetwork property
                if "subnetwork" in node and node["subnetwork"]:
                    if debug:
                        print(f"{'  ' * level}    Node {i} ({node_name}) has subnetwork")
                    current_track = collect_network_frames(
                        node["subnetwork"], current_track, f"{node_name} -> subnetwork", level + 1
                    )

                # 2. Node with nodes property (also a network)
                elif "nodes" in node and node["nodes"]:
                    if debug:
                        print(f"{'  ' * level}    Node {i} ({node_name}) has nodes property - treating as network")
                    current_track = collect_network_frames(node, current_track, node_name, level + 1)
                elif debug:
                    print(f"{'  ' * level}    Node {i} ({node_name}) has no sub-networks")

        return current_track

    # Start collecting from the root network
    total_tracks = collect_network_frames(network_dict, 0, network_name, 0)

    if debug:
        print(f"\nDebug Summary:")
        print(f"  Networks found: {networks_found[0]}")
        print(f"  Nodes checked: {nodes_checked[0]}")
        print(f"  Frames collected: {len(frames_data)}")

    # Calculate time bounds
    if frames_data:
        non_separator_frames = [f for f in frames_data if not f.get("is_separator")]
        if non_separator_frames:
            min_time = min(f["start"] for f in non_separator_frames)
            max_time = max(f["end"] for f in non_separator_frames)
            total_duration = max_time - min_time

            # Normalize times to start from 0
            for frame in non_separator_frames:
                frame["start"] -= min_time
                frame["end"] -= min_time
        else:
            total_duration = 0
            min_time = 0
            max_time = 0
    else:
        # No complete frames found
        raise ValueError(
            "No complete profiling frames found. All frames are missing end times, which suggests the profiling was not properly finalized."
        )

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LC Agent Profiling - {network.__class__.__name__}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            overflow: hidden;
        }}
        
        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        
        .header {{
            background: #2a2a2a;
            padding: 10px 20px;
            border-bottom: 1px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .title {{
            font-size: 18px;
            font-weight: 500;
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .control-label {{
            font-size: 12px;
            color: #999;
        }}
        
        button {{
            background: #3a3a3a;
            border: 1px solid #555;
            color: #e0e0e0;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        
        button:hover {{
            background: #4a4a4a;
        }}
        
        .main {{
            flex: 1;
            display: flex;
            position: relative;
            overflow: hidden;
        }}
        
        .timeline-container {{
            flex: 1;
            position: relative;
            overflow: hidden;
            cursor: grab;
        }}
        
        .timeline-container.dragging {{
            cursor: grabbing;
        }}
        
        .timeline {{
            position: absolute;
            width: 100%;
            min-height: 100%;
        }}
        
                 .track {{
             position: absolute;
             width: 100%;
             height: 30px;
             border-bottom: 1px solid #333;
             background: #242424;
         }}
         
         .track:nth-child(even) {{
             background: #282828;
         }}
        
        .frame {{
            position: absolute;
            height: 24px;
            top: 3px;
            border-radius: 3px;
            overflow: hidden;
            cursor: pointer;
            transition: opacity 0.2s;
            display: flex;
            align-items: center;
            font-size: 11px;
            white-space: nowrap;
            min-width: 0;  /* Allow very small frames */
            /* Default colors for unknown frame types */
            background: #666666;
            border: 1px solid #888888;
        }}
        
        .frame:hover {{
            opacity: 0.8;
            z-index: 10;
        }}
        
        .frame-network {{
            background: #4a7c8c;
            border: 1px solid #5a9cbc;
        }}
        
        .frame-node {{
            background: #6a5a8c;
            border: 1px solid #8a7abc;
        }}
        
        .frame-modifier {{
            background: #8c6a5a;
            border: 1px solid #bc8a7a;
        }}
        
        .frame-chunk {{
            background: #5a8c6a;
            border: 1px solid #7abc8a;
        }}
        
        .frame-custom {{
            background: #8c8c5a;
            border: 1px solid #bcbc7a;
        }}
        
        .frame-process_parents {{
            background: #5a7c8c;
            border: 1px solid #6a9cbc;
        }}
        
        .frame-combine_inputs {{
            background: #7c5a8c;
            border: 1px solid #9c6abc;
        }}
        
        .frame-retriever {{
            background: #8c5a7c;
            border: 1px solid #bc7a9c;
        }}
        
        .frame-text {{
            overflow: hidden;
            text-overflow: ellipsis;
            padding: 0 2px;
        }}
        
                 .separator {{
             position: absolute;
             width: 100%;
             height: 20px;
             background: #1e1e1e;
             border-bottom: 1px solid #333;
         }}
         
         .separator-text {{
             position: absolute;
             left: 0;
             display: inline-block;
             padding: 0 5px;
             font-size: 12px;
             color: #888;
             font-style: italic;
             line-height: 20px; /* Match separator height */
             transition: none; /* Disable transition for immediate updates */
         }}
        
        .tooltip {{
            position: fixed;
            background: #333;
            border: 1px solid #555;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            max-width: 400px;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }}
        
        .tooltip-header {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #fff;
        }}
        
        .tooltip-row {{
            margin: 2px 0;
            display: flex;
            justify-content: space-between;
            gap: 10px;
        }}
        
        .tooltip-label {{
            color: #999;
        }}
        
        .tooltip-value {{
            color: #fff;
            font-family: monospace;
        }}
        
        .time-ruler {{
            position: sticky;
            top: 0;
            height: 30px;
            background: #2a2a2a;
            border-bottom: 2px solid #444;
            z-index: 100;
        }}
        
        .time-mark {{
            position: absolute;
            top: 0;
            height: 100%;
            border-left: 1px solid #555;
            font-size: 10px;
            padding: 5px;
            color: #999;
        }}
        
        .stats {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(42, 42, 42, 0.9);
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            border: 1px solid #444;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">LC Agent Profiling - {network_name}</div>
            <div class="controls">
                <div class="control-group">
                    <span class="control-label">Zoom:</span>
                    <button onclick="zoom(1.5)">+</button>
                    <button onclick="zoom(0.67)">-</button>
                    <button onclick="resetZoom()">Reset</button>
                </div>
                <div class="control-group">
                    <span class="control-label">Total Duration:</span>
                    <span>{total_duration:.3f}s</span>
                </div>
            </div>
        </div>
        <div class="main">
            <div class="timeline-container" id="timeline-container">
                <div class="timeline" id="timeline">
                    <div class="time-ruler" id="time-ruler"></div>
                </div>
        </div>
            <div class="tooltip" id="tooltip" style="display: none;"></div>
        </div>
    </div>
    
    <script>
        const frames = {json.dumps(frames_data)};
        const totalDuration = {total_duration};
        const pixelsPerSecond = 1000; // Initial scale
        
        let currentScale = 1;
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartY = 0;
        
        const timeline = document.getElementById('timeline');
        const timelineContainer = document.getElementById('timeline-container');
        const tooltip = document.getElementById('tooltip');
        const timeRuler = document.getElementById('time-ruler');
        
        function createTimeRuler() {{
            timeRuler.innerHTML = '';
            const width = totalDuration * pixelsPerSecond * currentScale;
            timeRuler.style.width = width + 'px';
            
            // Calculate visible range
            const rect = timelineContainer.getBoundingClientRect();
            const viewportWidth = rect.width;
            const visibleStartTime = Math.max(0, (-offsetX) / (pixelsPerSecond * currentScale));
            const visibleEndTime = Math.min(totalDuration, (viewportWidth - offsetX) / (pixelsPerSecond * currentScale));
            const visibleDuration = visibleEndTime - visibleStartTime;
            
            // Calculate appropriate interval based on visible duration
            const interval = getTimeInterval(visibleDuration);
            
            // Start from a nice round number
            const startTime = Math.floor(visibleStartTime / interval) * interval;
            
            // Create marks for visible range plus some buffer
            for (let t = Math.max(0, startTime - interval * 2); t <= Math.min(totalDuration, visibleEndTime + interval * 2); t += interval) {{
                const mark = document.createElement('div');
                mark.className = 'time-mark';
                mark.style.left = (t * pixelsPerSecond * currentScale) + 'px';
                mark.textContent = formatTime(t);
                timeRuler.appendChild(mark);
            }}
        }}
        
        function getTimeInterval(visibleDuration) {{
            // Extended intervals to handle microseconds to hours
            const intervals = [
                0.000001, 0.000002, 0.000005,  // 1, 2, 5 microseconds
                0.00001, 0.00002, 0.00005,      // 10, 20, 50 microseconds
                0.0001, 0.0002, 0.0005,         // 100, 200, 500 microseconds
                0.001, 0.002, 0.005,            // 1, 2, 5 milliseconds
                0.01, 0.02, 0.05,               // 10, 20, 50 milliseconds
                0.1, 0.2, 0.5,                  // 100, 200, 500 milliseconds
                1, 2, 5,                        // 1, 2, 5 seconds
                10, 20, 30, 60,                 // 10, 20, 30, 60 seconds
                120, 300, 600,                  // 2, 5, 10 minutes
                1800, 3600                      // 30 minutes, 1 hour
            ];
            
            const targetMarks = 5; // Aim for 5-10 marks visible
            const maxMarks = 15;
            
            for (const interval of intervals) {{
                const marks = visibleDuration / interval;
                if (marks >= targetMarks && marks <= maxMarks) {{
                    return interval;
                }}
            }}
            
            // Fallback: divide visible duration by target marks
            return visibleDuration / targetMarks;
        }}
        
        function formatTime(seconds) {{
            if (seconds === 0) {{
                return '0';
            }} else if (seconds < 0.000001) {{
                return (seconds * 1000000000).toFixed(0) + 'ns';
            }} else if (seconds < 0.001) {{
                const us = seconds * 1000000;
                return us < 10 ? us.toFixed(1) + 'μs' : us.toFixed(0) + 'μs';
            }} else if (seconds < 1) {{
                const ms = seconds * 1000;
                return ms < 10 ? ms.toFixed(1) + 'ms' : ms.toFixed(0) + 'ms';
            }} else if (seconds < 60) {{
                return seconds < 10 ? seconds.toFixed(1) + 's' : seconds.toFixed(0) + 's';
            }} else if (seconds < 3600) {{
                const minutes = Math.floor(seconds / 60);
                const secs = seconds % 60;
                return secs > 0 ? `${{minutes}}m ${{secs.toFixed(0)}}s` : `${{minutes}}m`;
            }} else {{
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                return minutes > 0 ? `${{hours}}h ${{minutes}}m` : `${{hours}}h`;
            }}
        }}
        
                 function renderFrames() {{
             // Clear existing frames (except ruler)
             const existingFrames = timeline.querySelectorAll('.track, .separator');
             existingFrames.forEach(el => el.remove());
             
             // Find maximum track number
             let maxTrack = 0;
             frames.forEach(frame => {{
                 if (!frame.is_separator && frame.track !== undefined) {{
                     maxTrack = Math.max(maxTrack, frame.track);
                 }}
             }});
             
             // Create tracks and add frames
             let currentY = 30; // Start after ruler
             const separatorPositions = new Map();
             
             // First pass: find where separators should go
             frames.forEach(frame => {{
                 if (frame.is_separator) {{
                     separatorPositions.set(frame.track, frame.name);
                 }}
             }});
             
             // Create tracks with proper spacing
             for (let t = 0; t <= maxTrack; t++) {{
                 // Add separator before this track if needed
                 if (separatorPositions.has(t)) {{
                     const sep = document.createElement('div');
                     sep.className = 'separator';
                     sep.style.top = currentY + 'px';
                     
                     const sepText = document.createElement('div');
                     sepText.className = 'separator-text';
                     sepText.textContent = separatorPositions.get(t);
                     sep.appendChild(sepText);
                     
                     timeline.appendChild(sep);
                     currentY += 20; // Height of separator
                 }}
                 
                 // Create track
                 const track = document.createElement('div');
                 track.className = 'track';
                 track.style.top = currentY + 'px';
                 track.dataset.trackNum = t;
                 timeline.appendChild(track);
                 currentY += 30; // Height of track
             }}
             
             // Add frames to their respective tracks
             frames.forEach(frame => {{
                 if (!frame.is_separator) {{
                     const frameEl = createFrame(frame);
                     const track = timeline.querySelector(`[data-track-num="${{frame.track}}"]`);
                     if (track) {{
                         track.appendChild(frameEl);
                     }}
                 }}
             }});
             
             // Update timeline dimensions
             const width = totalDuration * pixelsPerSecond * currentScale;
             timeline.style.width = width + 'px';
             timeline.style.height = currentY + 'px';
         }}
        
                 function createFrame(frame) {{
             const el = document.createElement('div');
             el.className = 'frame frame-' + frame.type;
             const left = frame.start * pixelsPerSecond * currentScale;
             const width = frame.duration * pixelsPerSecond * currentScale;
             
             // Use transform for sub-pixel precision
             el.style.left = '0px';
             el.style.transform = `translateX(${{left}}px)`;
                         el.style.width = width + 'px';
            
            // Only show text if frame is wide enough
            if (width > 20) {{
                const text = document.createElement('div');
                text.className = 'frame-text';
                text.textContent = frame.name;
                el.appendChild(text);
            }}
             
             // Tooltip
             el.addEventListener('mouseenter', (e) => showTooltip(e, frame));
             el.addEventListener('mouseleave', hideTooltip);
             
             return el;
         }}
        
        function showTooltip(event, frame) {{
            let html = `<div class="tooltip-header">${{frame.name}}</div>`;
            html += `<div class="tooltip-row"><span class="tooltip-label">Type:</span><span class="tooltip-value">${{frame.type}}</span></div>`;
            html += `<div class="tooltip-row"><span class="tooltip-label">Duration:</span><span class="tooltip-value">${{frame.duration_str}}</span></div>`;
            html += `<div class="tooltip-row"><span class="tooltip-label">Start:</span><span class="tooltip-value">${{formatTime(frame.start)}}</span></div>`;
            html += `<div class="tooltip-row"><span class="tooltip-label">End:</span><span class="tooltip-value">${{formatTime(frame.end)}}</span></div>`;
            
            // Add metadata
            if (frame.metadata && Object.keys(frame.metadata).length > 0) {{
                html += '<div style="margin-top: 5px; border-top: 1px solid #555; padding-top: 5px;">';
                for (const [key, value] of Object.entries(frame.metadata)) {{
                    if (key === 'content' && value) {{
                        // Truncate long content
                        const content = value.length > 100 ? value.substring(0, 100) + '...' : value;
                        html += `<div class="tooltip-row"><span class="tooltip-label">${{key}}:</span></div>`;
                        html += `<div style="margin-left: 10px; font-family: monospace; color: #aaa;">${{content}}</div>`;
                    }} else {{
                        html += `<div class="tooltip-row"><span class="tooltip-label">${{key}}:</span><span class="tooltip-value">${{value}}</span></div>`;
                    }}
                }}
                html += '</div>';
            }}
            
            tooltip.innerHTML = html;
            tooltip.style.display = 'block';
            updateTooltipPosition(event);
        }}
        
        function hideTooltip() {{
            tooltip.style.display = 'none';
        }}
        
        function updateTooltipPosition(event) {{
            const x = event.clientX + 10;
            const y = event.clientY + 10;
            
            // Adjust if tooltip goes off screen
            const rect = tooltip.getBoundingClientRect();
            const adjustedX = x + rect.width > window.innerWidth ? x - rect.width - 20 : x;
            const adjustedY = y + rect.height > window.innerHeight ? y - rect.height - 20 : y;
            
            tooltip.style.left = adjustedX + 'px';
            tooltip.style.top = adjustedY + 'px';
        }}
        
        function zoom(factor) {{
            // Calculate center of viewport
            const rect = timelineContainer.getBoundingClientRect();
            const centerX = rect.width / 2;
            
            // Calculate the point on the timeline at viewport center
            const timelineX = (centerX - offsetX) / currentScale;
            
            // Apply zoom
            currentScale *= factor;
            currentScale = Math.max(0.1, Math.min(10, currentScale));
            
            // Calculate new offset to keep the same timeline point at center
            // Only adjust X offset for horizontal zooming
            offsetX = centerX - timelineX * currentScale;
            
            updateView();
        }}
        
        function resetZoom() {{
            currentScale = 1;
            offsetX = 0;
            offsetY = 0;
            updateView();
        }}
        
        function updateView() {{
            timeline.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px)`;
            createTimeRuler();
            renderFrames();
            updateSeparatorTextPositions();
        }}
        
        function updateSeparatorTextPositions() {{
            // Update all separator text positions to compensate for scroll
            const separatorTexts = timeline.querySelectorAll('.separator-text');
            separatorTexts.forEach(text => {{
                // Calculate desired position (10px from viewport left)
                const desiredX = -offsetX + 10;
                
                // But constrain it to stay within the separator bounds
                // Don't let it go to the left of the separator start (0)
                const constrainedX = Math.max(0, desiredX);
                
                // Also don't let it go beyond the right edge of the viewport
                // when the timeline is very narrow
                const timelineWidth = timeline.offsetWidth;
                const viewportWidth = timelineContainer.offsetWidth;
                const maxX = Math.min(timelineWidth - 100, viewportWidth - offsetX - 100);
                
                text.style.transform = `translateX(${{Math.min(constrainedX, maxX)}}px)`;
            }});
        }}
        
                 // Dragging
         timelineContainer.addEventListener('mousedown', (e) => {{
            isDragging = true;
            dragStartX = e.clientX - offsetX;
            dragStartY = e.clientY - offsetY;
             timelineContainer.classList.add('dragging');
             e.preventDefault();
         }});
        
        document.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                offsetX = e.clientX - dragStartX;
                offsetY = e.clientY - dragStartY;
                offsetY = Math.min(0, offsetY); // Don't allow dragging above the top
                timeline.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px)`;
                
                // Update time ruler and separator positions while dragging
                createTimeRuler();
                updateSeparatorTextPositions();
            }}
            
            if (tooltip.style.display === 'block') {{
                updateTooltipPosition(e);
            }}
        }});
        
        document.addEventListener('mouseup', () => {{
            isDragging = false;
            timelineContainer.classList.remove('dragging');
        }});
        
        // Mouse wheel zoom
        timelineContainer.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            
            // Get mouse position relative to the timeline container
            const rect = timelineContainer.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            
            // Calculate the point on the timeline under the mouse before zoom
            const timelineX = (mouseX - offsetX) / currentScale;
            
            // Apply zoom
            currentScale *= factor;
            currentScale = Math.max(0.01, Math.min(100, currentScale));
            
            // Calculate new offset to keep the same timeline point under the mouse
            // Only adjust X offset, keep Y offset unchanged
            offsetX = mouseX - timelineX * currentScale;
            
            updateView();
        }});
        
        // Initial render
        updateView();
    </script>
</body>
</html>"""
    
    # Write to file
    output_path = Path(output_path)
    output_path.write_text(html_content, encoding="utf-8")

    return str(output_path.absolute())
